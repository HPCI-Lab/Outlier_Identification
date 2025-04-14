from prov4ml.utils.prov_getters import *
import json
import os
from configs.run_configs import RunConfig
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH, DST_CSV_PATH
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from detection.common import compute_metrics, get_metrics_from_config, get_filtered_metric, plot_f1_chart

def compute_expma(d, configs : RunConfig, alpha): 
    if configs.detection.use_pca: 
        pca = PCA(n_components=1)
        d = pd.DataFrame(d, index=range(len(d)))
        d = pca.fit_transform(d).squeeze()

    ewm_mean = pd.Series(d).ewm(alpha=alpha).mean()
    ewm_std = pd.Series(d).ewm(alpha=alpha).std()
    
    y_pred = abs(d - ewm_mean) > 5 * ewm_std
    return [1 if y else 0 for y in y_pred]



def compute_accuracy(configs : RunConfig):
    n = len(os.listdir("./prov/"))-1
    file_path = f"./prov/IBM_outliers_{n}/provgraph_IBM_outliers.json"
    
    data = json.load(open(file_path))
    loss = get_metric(data, "Loss_Context.TRAINING", sort_by="time")
    loss = loss[loss["epoch"].isin(configs.detection.keep_epochs)]["value"] if configs.detection.keep_epochs != [] else loss["value"]

    gt = get_metric(data, "Outlier_Context.TRAINING", sort_by="time")
    gt = gt[gt["epoch"].isin(configs.detection.keep_epochs)]["value"] if configs.detection.keep_epochs != [] else gt["value"]
    gt = [any(eval(g)) for g in gt]

    LOF = compute_expma(loss.values, configs, alpha=0.1)
    print("EXPMA")
    print(pd.Series(LOF).value_counts())
    print(compute_metrics(gt, LOF))


def compute_bruteforce_accuracy(configs : RunConfig, create_chart=False, id_exp=None): 
    window_sizes = np.arange(0.1, 1.0, 0.05)
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
    gt = [1 if any(eval(g)) else 0 for g in gt.to_numpy()]

    res = []
    for w in window_sizes: 
        configs.detection.window_size = w
        HAM = compute_expma(comb.values, configs, alpha=w)

        p, r, f, _ = precision_recall_fscore_support(gt, HAM, average="binary")

        res.append({
            "window_size": w, 
            "precision": p, 
            "recall": r, 
            "f1_score": f, 
        })

    res = pd.DataFrame(res, index=range(len(res)))
    dstfilename = f"EXPMA_{configs.model.type}_{configs.dataset.type}_{configs.dataset.outlier_percentage}"
    res.to_csv(f"{DST_CSV_PATH}/{dstfilename}.csv")

    if create_chart: 
        plot_f1_chart(res, dstfilename)