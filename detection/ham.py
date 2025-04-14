import json
import hampel
import os
import pandas as pd
from prov4ml.utils.prov_getters import *
from detection.common import compute_metrics
from sklearn.decomposition import PCA
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH, DST_CSV_PATH
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from detection.common import compute_metrics, get_metrics_from_config, get_filtered_metric, plot_f1_chart

def compute_HAM(d, configs : RunConfig): 
    if configs.detection.use_pca: 
        pca = PCA(n_components=1)
        d = pd.DataFrame(d, index=range(len(d)))
        d = pca.fit_transform(d).squeeze()

    result = hampel.hampel(d, configs.detection.window_size)
    indices = []
    for i in range(len(result.medians)): 
        indices.append(1 if i in result.outlier_indices else 0)
    return indices, result.medians


def compute_accuracy(configs : RunConfig):
    n = len(os.listdir(f"{CACHE_PATH}/prov/"))-1
    file_path = f"{CACHE_PATH}/prov/IBM_outliers_{n}/provgraph_IBM_outliers.json"
    
    data = json.load(open(file_path))
    comb = get_metrics_from_config(configs, data)

    gt = get_filtered_metric(configs, data, "Outlier")
    gt = [1 if any(eval(g)) else 0 for g in gt]

    LOF, _ = compute_HAM(comb.values, configs)
    print("HAM")
    print(pd.Series(LOF).value_counts())
    print(compute_metrics(gt, LOF))



def compute_bruteforce_accuracy(configs : RunConfig, create_chart=False, id_exp=None): 
    window_sizes = [2,3,4,5,7,9,11, 20, 30, 40, 50, 75, 100]
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
        HAM, _ = compute_HAM(comb.values, configs)
        # gt, HAM = np.array(gt), np.array(HAM)

        p, r, f, _ = precision_recall_fscore_support(gt, HAM, average="binary")

        res.append({
            "window_size": w, 
            "precision": p, 
            "recall": r, 
            "f1_score": f, 
        })

    res = pd.DataFrame(res, index=range(len(res)))
    dstfilename = f"HAM_{configs.model.type}_{configs.dataset.type}_{configs.dataset.outlier_percentage}_bs{configs.dataset.batch_size}"
    res.to_csv(f"{DST_CSV_PATH}/{dstfilename}.csv")

    if create_chart: 
        plot_f1_chart(res, dstfilename)
