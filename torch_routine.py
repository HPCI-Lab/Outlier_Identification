
import argparse
import prov4ml
import time
import torch
import os
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from utils.metrics import calculate_metrics
from configs.getters import * 
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH
from torch_distributed_helper import *


def train_model(configs : RunConfig, model, criterion, optimizer, train_loader):
    model.train()
    for e in range(configs.run.epochs):
        for indices, is_outlier, X_batch, y_batch in tqdm(train_loader):
            step_time = time.time()
            # optimizer.zero_grad()

            X_batch, y_batch = X_batch.to(configs.run.device), y_batch.to(configs.run.device)

            # outputs, loss = model(X_batch)
            # if loss is None: 
            #     loss = criterion(outputs, y_batch)

            loss = model((X_batch, y_batch))

            backward_time = time.time()
            # loss.backward()
            model.backward(loss)
    
            optim_time = time.time()
            # optimizer.step()
            model.step()
    
            end_time = time.time()

            prov4ml.log_metric("Step_time", end_time - step_time, prov4ml.Context.TRAINING, step=e)
            prov4ml.log_metric("Backward_time", end_time - backward_time, prov4ml.Context.TRAINING, step=e)
            prov4ml.log_metric("Optim_step_time", end_time - optim_time, prov4ml.Context.TRAINING, step=e)
            prov4ml.log_metric("Indices", indices.tolist(), prov4ml.Context.TRAINING, step=e)
            prov4ml.log_metric("Loss", loss.item(), prov4ml.Context.TRAINING, step=e)
            prov4ml.log_metric("Outlier", is_outlier.tolist(), prov4ml.Context.TRAINING, step=e)


def main(configs : RunConfig): 

    os.environ['HF_HOME'] = CACHE_PATH
    os.environ['HF_DATASETS_CACHE'] = CACHE_PATH

    setup()

    torch.set_default_dtype(torch.bfloat16)

    prov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name="IBM_outliers", 
        provenance_save_dir=f"{CACHE_PATH}/prov",
        save_after_n_logs=100,
        collect_all_processes=False, 
        disable_codecarbon = True, 
    )

    # prov4ml.log_artifact("./IBM/config.yaml", prov4ml.Context.TRAINING)

    dataset = get_dataset(configs)
    # train_loader = get_dataloader(configs, dataset)
    train_loader = to_distributed_dataloader(dataset, batch_size=configs.dataset.batch_size)

    model = get_model(configs).to(configs.run.device)
    # model = FSDP(model) #auto_wrap_policy=size_based_auto_wrap_policy,

    model, optimizer, _, _ = deepspeed.initialize(model=model)

    criterion = get_criterion(configs).to(configs.run.device)
    optimizer = get_optimizer(configs, model)

    train_model(configs, model, criterion, optimizer, train_loader)

    prov4ml.end_run()

    cleanup()

    # calculate_metrics(configs)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf')  
    args = parser.parse_args()

    configs = RunConfig(args.conf)
    main(configs)