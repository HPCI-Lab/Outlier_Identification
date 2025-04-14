
import numpy as np
import pandas as pd
from configs.run_configs import RunConfig
from configs.paths import DST_IMG_PATH
from prov4ml.utils.prov_getters import *
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def get_metrics_from_config(configs : RunConfig, data): 
    total = []
    for metric in configs.identification.metrics: 
        m = get_filtered_metric(configs, data, metric)
        total.append(m)
    return pd.concat(total, axis=1)
    
def get_filtered_metric(configs : RunConfig, data, metric): 
    e = get_metric(data, f"Loss_Context.TRAINING", sort_by="time")["epoch"]

    m = get_metric(data, f"{metric}_Context.TRAINING", sort_by="time")
    m["epoch"] = e
    m = m[m["epoch"].isin(configs.identification.keep_epochs)]["value"] if configs.identification.keep_epochs != [] else m["value"]
    return m

def get_filtered_metric_full(configs : RunConfig, data, metric): 
    m = get_metric(data, f"{metric}_Context.TRAINING", sort_by="time")
    m = m[m["epoch"].isin(configs.identification.keep_epochs)] if configs.identification.keep_epochs != [] else m
    return m

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return pd.DataFrame({
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }, index=[0]).T.round(3)

def plot_f1_chart(res, dstfilename): 
    sns.lineplot(res, x="window_size", y="precision", color="tab:blue", label="precision")
    sns.lineplot(res, x="window_size", y="recall", color="tab:orange", label="recall")
    sns.lineplot(res, x="window_size", y="f1_score", color="tab:green", label="f1_score")
    plt.ylabel("")
    plt.legend()
    plt.savefig(f"{DST_IMG_PATH}/{dstfilename}.png")
    plt.clf()

def get_epochs(configs): 
    if configs.identification.keep_epochs != []: 
        return configs.identification.keep_epochs 
    else:  
        return range(configs.run.epochs)

def get_value_counts_epoch_threshold(configs): 
    if configs.identification.keep_epochs != []: 
        return len(configs.identification.keep_epochs)-1
    else: 
        return configs.run.epochs -1
