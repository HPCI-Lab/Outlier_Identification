
import argparse
import os

from utils.metrics import calculate_metrics
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH

import matplotlib.pyplot as plt
from identification.common import get_filtered_metric
import json

def main(configs : RunConfig, id_exp : int): 

    os.environ['HF_HOME'] = CACHE_PATH
    os.environ['HF_DATASETS_CACHE'] = CACHE_PATH
    os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH

    if id_exp is None: 
        id_exp = len(os.listdir(f"{CACHE_PATH}/prov/"))-1

    old = configs.identification.keep_epochs
    file_path = f"{CACHE_PATH}/prov/IBM_outliers_{id_exp}/provgraph_IBM_outliers.json"
    data = json.load(open(file_path))
    configs.identification.keep_epochs = []
    gt = get_filtered_metric(configs, data, "Loss")
    outs = [any(eval(v)) for v in get_filtered_metric(configs, data, "Outlier").tolist()]
    plt.plot(gt)
    for i in range(len(outs)):
        if outs[i]:
            plt.axvline(i, color='red', linestyle='dashed', alpha=0.7, linewidth=1)

    plt.savefig(f"loss_{id_exp}")
    plt.clf()

    configs.identification.keep_epochs = old
    calculate_metrics(configs, id_exp)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf')  
    parser.add_argument('-i', '--id')  
    args = parser.parse_args()

    configs = RunConfig(args.conf)
    main(configs, args.id if hasattr(args, "id") else None)