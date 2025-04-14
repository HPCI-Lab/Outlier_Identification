import json
import pandas as pd
import os
from prov4ml.utils.prov_getters import *
from sklearn.neighbors import LocalOutlierFactor
from detection.common import compute_metrics
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH, DST_CSV_PATH
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from detection.common import compute_metrics, get_metrics_from_config, get_filtered_metric, plot_f1_chart
from sklearn import preprocessing

def compute_LOF(d, configs : RunConfig, contamination="auto"): 
    if configs.detection.use_pca: 
        pca = PCA(n_components=2)
        d = pd.DataFrame(d, index=range(len(d)))
        d = pca.fit_transform(d).squeeze()

    clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    y_pred = clf.fit_predict(d)
    return [1 if y == -1 else 0 for y in y_pred], clf.negative_outlier_factor_


def compute_accuracy(configs : RunConfig):
    n = len(os.listdir(f"{CACHE_PATH}/prov/"))-1
    file_path = f"{CACHE_PATH}/prov/IBM_outliers_{n}/provgraph_IBM_outliers.json"
    
    data = json.load(open(file_path))
    comb = get_metrics_from_config(configs, data)

    gt = get_filtered_metric(configs, data, "Outlier")
    gt = [1 if any(eval(g)) else 0 for g in gt]

    LOF, _ = compute_LOF(comb.values, configs)
    print("LOF")
    print(pd.Series(LOF).value_counts())
    print(compute_metrics(gt, LOF))


def compute_bruteforce_accuracy(configs : RunConfig, create_chart=False, id_exp=None): 
    window_sizes = np.arange(0.1, 0.5, 0.05)
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
        LOF, _ = compute_LOF(comb.values, configs, contamination=w)
        p, r, f, s = precision_recall_fscore_support(gt, LOF, average="binary")
        res.append({
            "window_size": w, 
            "precision": p, 
            "recall": r, 
            "f1_score": f, 
        })

    res = pd.DataFrame(res, index=window_sizes)
    dstfilename = f"LOF_{configs.model.type}_{configs.dataset.type}_{configs.dataset.outlier_percentage}_bs{configs.dataset.batch_size}"
    res.to_csv(f"{DST_CSV_PATH}/{dstfilename}.csv")

    if create_chart: 
        plot_f1_chart(res, dstfilename)