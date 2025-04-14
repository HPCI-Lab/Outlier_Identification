import json
import os
from prov4ml.utils.prov_getters import *
from detection.wlf import compute_wlf
from detection.mahalanobis import compute_mahalanobis
from detection.lof import compute_LOF
from identification.common import get_epochs
from detection.common import get_metrics_from_config, get_filtered_metric, get_filtered_metric_full
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing


def multi_list_intersection(batches_of_indices):
    bf = set(batches_of_indices[0])
    for i in range(1, len(batches_of_indices)): 
        bf = bf.intersection(set(batches_of_indices[i]))
    return list(bf)
    # return list(set.intersection(*map(set,batches_of_indices)))

def get_common_outliers(configs : RunConfig, id_exp=None):
    if id_exp is None: 
        id_exp = len(os.listdir(f"{CACHE_PATH}/prov/"))-1

    file_path = f"{CACHE_PATH}/prov/IBM_outliers_{id_exp}/provgraph_IBM_outliers.json"

    data = json.load(open(file_path))
    comb = get_metrics_from_config(configs, data)
    cols = range(len(comb.columns ))
    comb.columns = cols
    scaler = preprocessing.StandardScaler()
    comb = scaler.fit_transform(comb)
    comb = pd.DataFrame(comb, columns=cols)

    gt = get_filtered_metric(configs, data, "Outlier")
    gt = [1 if any(eval(g)) else 0 for g in gt.to_numpy()]
    indices = get_filtered_metric_full(configs, data, "Indices").reset_index()

    print("WLF FOR SUBSETTING")
    outliers = compute_wlf(comb.values, configs, alpha=0.8)
    outliers = pd.Series(outliers)
    calculate_value_counting(configs, outliers, indices, gt)

    print("MAHA FOR SUBSETTING")
    outliers = compute_mahalanobis(comb.values, configs, confidence=0.2)
    outliers = pd.Series(outliers)
    calculate_value_counting(configs, outliers, indices, gt)

    print("LOF FOR SUBSETTING")
    outliers, _ = compute_LOF(comb.values, configs, contamination=0.4)
    outliers = pd.Series(outliers)
    calculate_value_counting(configs, outliers, indices, gt)


def calculate_value_counting(configs : RunConfig, outliers_pred, indices, gt_outliers): 
    possible_indices = [[]] * len(get_epochs(configs))
    for i, e in enumerate(get_epochs(configs)): 
        outliers_e = outliers_pred[indices["epoch"] == e]
        indices_e = indices[indices["epoch"] == e]
        for o in range(len(outliers_e)): 
            if outliers_e.iloc[o] == 1: 
                possible_indices[i] += eval(indices_e["value"].iloc[o])
    
    possible_indices = multi_list_intersection(possible_indices)
    pred = [1 if i in possible_indices else 0 for i in range(len(gt_outliers))]

    print("F1:\t", precision_recall_fscore_support(gt_outliers, pred, average="binary"))