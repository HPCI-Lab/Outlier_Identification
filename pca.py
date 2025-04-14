
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from local_datasets.imdb import ImdbReviewsDataset
from local_datasets.med_magic import HealthCareMagicDataset, HealthCareMagicChatDataset
from local_datasets.chat_doctor import ChatDoctorDataset
from configs.run_configs import RunConfig
import torch
from sklearn.manifold import TSNE

c = RunConfig("/u/lelepado/IBM/c_granite.yaml")
c.dataset.samples_per_class = 5000
df = HealthCareMagicDataset(c)
# df = ImdbReviewsPizzaOutliersDataset(RunConfig("/u/lelepado/IBM/c_granite.yaml"))


device = "cuda"
model_path = "ibm-granite/granite-3.1-2b-base"

ls = [data.unsqueeze(0) for _, _, _, data in df]
output_tokens = torch.concat(ls, dim=0)

ls = [data["input_ids"].float() for _, _, data, _ in df]
input_tokens = torch.concat(ls, dim=0)
input_tokens[14] += 5000000

# pca = PCA(n_components=2)
# data_pca1 = pca.fit_transform(input_tokens)

# pca = PCA(n_components=2)
# data_pca2 = pca.fit_transform(output_tokens)

tsne = TSNE(n_components=2, perplexity=3, random_state=42)
data_pca1 = tsne.fit_transform(input_tokens)

tsne = TSNE(n_components=2, perplexity=3, random_state=42)
data_pca2 = tsne.fit_transform(output_tokens)

# iso_forest = IsolationForest(contamination=0.001)  # adjust contamination as needed
# outliers = iso_forest.fit_predict(data_pca)

# plt.figure(figsize=(10, 6))
# plt.scatter(data_pca[:, 0], data_pca[:, 1], c=outliers, cmap='coolwarm', edgecolor='k')

# outlier_indices = np.where(outliers == -1)
# plt.scatter(data_pca[outlier_indices, 0], data_pca[outlier_indices, 1], color='red', s=100, label='Outliers')
# plt.title("PCA of Encoded Text Data with Outliers")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend()
# plt.savefig("pca_med_ds.png")
# plt.clf()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

f.suptitle("PCA of Encoded Text HealthCareMagicDataset")
f.set_figheight(15)
f.set_figwidth(30)
ax1.scatter(data_pca1[:, 0], data_pca1[:, 1], c="blue", edgecolor='k')
ax1.set_xlabel("Input Data")
ax1.set_ylabel("PCA Component")
ax2.scatter(data_pca2[:, 0], data_pca2[:, 1], c='red', edgecolor='k')
ax2.set_xlabel("Output Data")
ax2.set_ylabel("PCA Component")
plt.tight_layout()

f.savefig(f"pca_{type(df)}.png")
