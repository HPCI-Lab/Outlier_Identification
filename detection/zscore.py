from prov4ml.utils.prov_getters import *
from scipy.stats import zscore
import numpy as np
import json
import os
import pandas as pd
from detection.common import compute_metrics, get_metrics_from_config, get_filtered_metric, plot_f1_chart
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH, DST_CSV_PATH
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support

def compute_zscore(d, configs : RunConfig, degs=3): 
    if configs.detection.use_pca: 
        pca = PCA(n_components=1)
        d = pd.DataFrame(d, index=range(len(d)))
        d = pca.fit_transform(d).squeeze()

    y_pred = zscore(d)
    y_pred = np.abs(y_pred) < degs
    return [0 if y else 1 for y in y_pred]


def compute_accuracy(configs : RunConfig):
    n = len(os.listdir("./prov/"))-1
    file_path = f"./prov/IBM_outliers_{n}/provgraph_IBM_outliers.json"
    
    data = json.load(open(file_path))
    loss = get_metric(data, "Loss_Context.TRAINING", sort_by="time")
    loss = loss[loss["epoch"].isin(configs.detection.keep_epochs)]["value"] if configs.detection.keep_epochs != [] else loss["value"]

    gt = get_metric(data, "Outlier_Context.TRAINING", sort_by="time")
    gt = gt[gt["epoch"].isin(configs.detection.keep_epochs)]["value"] if configs.detection.keep_epochs != [] else gt["value"]
    gt = [any(eval(g)) for g in gt]

    LOF = compute_zscore(loss.values, configs)
    print("ZSCORE")
    print(pd.Series(LOF).value_counts())
    print(compute_metrics(gt, LOF))


def compute_bruteforce_accuracy(configs : RunConfig, create_chart=False, id_exp=None): 
    window_sizes = np.arange(1.0, 3.0, 0.1)
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
        LOF = compute_zscore(comb.values, configs, degs=w)
        p, r, f, _ = precision_recall_fscore_support(gt, LOF, average="binary")
        res.append({
            "window_size": w, 
            "precision": p, 
            "recall": r, 
            "f1_score": f
        })
    res = pd.DataFrame(res, index=window_sizes)

    dstfilename = f"ZSCORE_{configs.model.type}_{configs.dataset.type}_{configs.dataset.outlier_percentage}"
    res.to_csv(f"{DST_CSV_PATH}/{dstfilename}.csv")

    if create_chart: 
        plot_f1_chart(res, dstfilename)