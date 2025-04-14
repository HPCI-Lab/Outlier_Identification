import json
import numpy as np
import os
from prov4ml.utils.prov_getters import *
from detection.common import get_metrics_from_config, get_filtered_metric
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from detection.mahalanobis import compute_mahalanobis
from detection.lof import compute_LOF
from detection.wlf import compute_wlf
from detection.zscore import compute_zscore

def compute_accuracy(configs : RunConfig, OUTLIER_FP_RATIO):
    n = len(os.listdir(f"{CACHE_PATH}/prov/"))-1
    file_path = f"{CACHE_PATH}/prov/IBM_outliers_{n}/provgraph_IBM_outliers.json"
    
    data = json.load(open(file_path))
    comb = get_metrics_from_config(configs, data)

    gt = get_filtered_metric(configs, data, "Outlier")
    gt = [1 if any(eval(g)) else 0 for g in gt]

    MAHA = compute_mahalanobis(comb.values, configs, confidence=0.1)
    LOF, _ = compute_LOF(comb.values, configs, contamination=0.4)
    WLF = compute_wlf(comb.values, configs, alpha=0.8)
    ZSCORE = compute_zscore(comb.values, configs, degs=1)

    voting = np.array(MAHA) + np.array(LOF) + np.array(WLF) + np.array(ZSCORE)

    num_to_keep = int(configs.dataset.outlier_percentage * configs.dataset.samples_per_class * configs.dataset.classes * OUTLIER_FP_RATIO)
    voting = pd.Series(voting).sort_values().head(num_to_keep).index.to_list()
    pred = [1 if i in voting else 0 for i in range(len(gt))]

    print("VOTING")
    print(pd.Series(pred).value_counts())
    print(precision_recall_fscore_support(gt, pred, average="binary"))

