import json
import pandas as pd
import os
from prov4ml.utils.prov_getters import *
from configs.run_configs import RunConfig
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

class Autoencoder(nn.Module):  
	def __init__(self, original_size, hidden_size):
		super(Autoencoder, self).__init__()
		mid_size = (original_size + hidden_size) // 2
		self.fc1 = nn.Linear(original_size, mid_size)
		self.fc2 = nn.Linear(mid_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, mid_size)
		self.fc4 = nn.Linear(mid_size, original_size)

	def encode(self, x): 
		x = x.reshape(1, -1)
		z = torch.tanh(self.fc1(x))
		z = torch.tanh(self.fc2(z))  # latent in [-1,+1]
		return z  

	def decode(self, x): 
		z = torch.tanh(self.fc3(x))
		z = torch.sigmoid(self.fc4(z))  # [0.0, 1.0]
		return z
		
	def forward(self, x):
		z = x.reshape(1, -1)
		z = self.encode(z) 
		z = self.decode(z) 
		z = z.reshape(x.shape)
		return z  # in [0.0, 1.0]



def compute_accuracy(configs : RunConfig):
	n = len(os.listdir("./prov/"))-1
	file_path = f"./prov/IBM_outliers_{n}/provgraph_IBM_outliers.json"
	
	data = json.load(open(file_path))
	loss = get_metric(data, "Loss_Context.TRAINING", sort_by="time")
	# grad0 = get_metric(data, "Backward_time_Context.TRAINING", sort_by="time")
	grad1 = get_metric(data, "Step_time_Context.TRAINING", sort_by="time")
	grad2 = get_metric(data, "gpu_usage_Context.TRAINING", sort_by="time")
	gt = get_metric(data, "Outlier_Context.TRAINING", sort_by="time")

	if configs.detection.keep_epochs != []: 
		loss = loss[loss["epoch"].isin(configs.detection.keep_epochs)]
		# grad0 = grad0[grad0["epoch"].isin(configs.detection.keep_epochs)]
		grad1 = grad1[grad1["epoch"].isin(configs.detection.keep_epochs)]
		grad2 = grad2[grad2["epoch"].isin(configs.detection.keep_epochs)]
		gt = gt[gt["epoch"].isin(configs.detection.keep_epochs)]

	X = []
	eps = configs.detection.keep_epochs if configs.detection.keep_epochs != [] else range(int(max(loss["epoch"]))+1)
	for epoch in eps: 
		loss_e = loss[loss["epoch"] == epoch]["value"]
		if epoch == 0: 
			num_batches = len(loss_e)
		# grad0_e = grad0[grad0["epoch"] == epoch]["value"].values
		grad1_e = grad1[grad1["epoch"] == epoch]["value"]
		grad2_e = grad2[grad2["epoch"] == epoch]["value"]
		# gt_e = gt[gt["epoch"] == epoch]["value"]
		# gt_e = [1.0 if any(eval(g)) else 0.0 for g in gt_e]
		comb = pd.concat([loss_e, grad1_e, grad2_e], axis=1).values
		X.append(torch.tensor(comb).unsqueeze(0))

	dataset = TensorDataset(torch.concat(X).float())
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

	autoencoder = Autoencoder(3 * num_batches, num_batches)
	optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
	loss_fn = nn.MSELoss()

	num_epochs = 5
	for _ in tqdm(range(num_epochs)):
		for data in dataloader:
			optimizer.zero_grad()
			pred = autoencoder(data[0])

			loss = loss_fn(pred, data[0])
			loss.backward()
			optimizer.step()


	autoencoder.eval()
	latent_vectors = []
	with torch.no_grad():
		for data in dataloader:
			latent = autoencoder.encode(data[0]).squeeze().numpy().tolist()
			latent_vectors.extend(latent)
	latent_vectors = np.array(latent_vectors).reshape(-1, 1)

	is_outlier = np.array([1 if any(eval(g)) else -1 for g in gt["value"]])

	from sklearn.svm import OneClassSVM
	ocsvm = OneClassSVM(kernel='rbf', gamma='auto').fit(latent_vectors)
	predictions = ocsvm.predict(latent_vectors)
	print("Acc: ", accuracy_score(predictions, is_outlier))
	print("F1: ", f1_score(predictions, is_outlier))

	from sklearn.covariance import EllipticEnvelope
	clf = EllipticEnvelope(support_fraction=0.8, contamination=0.01) 
	clf.fit(latent_vectors)
	outliers = clf.predict(latent_vectors)
	print("Acc: ", accuracy_score(outliers, is_outlier))
	print("F1: ", f1_score(outliers, is_outlier))

	
