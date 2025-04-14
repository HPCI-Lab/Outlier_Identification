import json
from scipy.stats import chi2
import numpy as np
import os
from prov4ml.utils.prov_getters import *
from detection.common import compute_metrics, get_metrics_from_config, get_filtered_metric, plot_f1_chart
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH, DST_CSV_PATH
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support

def compute_mahalanobis(d, configs : RunConfig, confidence=0.99): 
    if configs.detection.use_pca: 
        pca = PCA(n_components=2)
        d = pd.DataFrame(d, index=range(len(d)))
        d = pca.fit_transform(d).squeeze()

    cov_matrix = np.cov(d.T)  # Covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)  # Inverse covariance matrix

    # Step 2: Compute Mahalanobis distance for each row
    mean_values = d.mean(axis=0)  # Mean of each variable
    mahalanobis_distances = []
    for i in range(len(d)):
        diff = d[i] - mean_values
        distance = np.sqrt(diff.T @ inv_cov_matrix @ diff)
        mahalanobis_distances.append(distance)

    # Step 3: Set threshold based on chi-squared distribution
    threshold = chi2.ppf(confidence, d.shape[1])  # 99% confidence level, degrees of freedom = number of variables
    outlier = mahalanobis_distances > threshold
    return [1 if y else 0 for y in outlier]


def compute_accuracy(configs : RunConfig):
    n = len(os.listdir(f"{CACHE_PATH}/prov/"))-1
    file_path = f"{CACHE_PATH}/prov/IBM_outliers_{n}/provgraph_IBM_outliers.json"
    
    data = json.load(open(file_path))
    comb = get_metrics_from_config(configs, data)

    gt = get_filtered_metric(configs, data, "Outlier")
    gt = [1 if any(eval(g)) else 0 for g in gt]

    LOF = compute_mahalanobis(comb.values, configs)
    print("MAHA")
    print(pd.Series(LOF).value_counts())
    print(compute_metrics(gt, LOF))



def compute_bruteforce_accuracy(configs : RunConfig, create_chart=False, id_exp=None): 
    window_sizes = np.arange(0.0, 1.0, 0.05)
    if id_exp is None: 
        id_exp = len(os.listdir(f"{CACHE_PATH}/prov/"))-1
    file_path = f"{CACHE_PATH}/prov/IBM_outliers_{id_exp}/provgraph_IBM_outliers.json"

    data = json.load(open(file_path))
    comb = get_metrics_from_config(configs, data)
    comb.columns = range(len(comb.columns ))
    scaler = preprocessing.StandardScaler()
    comb = scaler.fit_transform(comb)
    comb = pd.DataFrame(comb, columns=range(comb.shape[1]))

    gt = get_filtered_metric(configs, data, "Outlier")
    gt = [1 if any(eval(g)) else 0 for g in gt]

    res = []
    for w in window_sizes: 
        LOF = compute_mahalanobis(comb.values, configs, confidence=w)
        p, r, f, _ = precision_recall_fscore_support(gt, LOF, average="binary")
        res.append({
            "window_size": w, 
            "precision": p, 
            "recall": r, 
            "f1_score": f
        })

    res = pd.DataFrame(res, index=window_sizes)
    dstfilename = f"MAHA_{configs.model.type}_{configs.dataset.type}_{configs.dataset.outlier_percentage}_bs{configs.dataset.batch_size}"
    res.to_csv(f"{DST_CSV_PATH}/{dstfilename}.csv")

    if create_chart: 
        plot_f1_chart(res, dstfilename)