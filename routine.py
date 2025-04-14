
import argparse
from tqdm import tqdm
import prov4ml
import time
import torch 
import pynvml
import os

from utils.gradient import compute_gradient_norm_per_layer
from utils.metrics import calculate_metrics
from configs.getters import * 
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH

def train_model(configs : RunConfig, model, criterion, optimizer, train_loader):
    model.train()
    for e in range(configs.run.epochs):
        for indices, is_outlier, X_batch, y_batch in tqdm(train_loader):
            step_time = time.time()
            optimizer.zero_grad()

            X_batch, y_batch = X_batch.to(configs.run.device), y_batch.to(configs.run.device)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()
            end_time = time.time()
            prov4ml.log_metric(f"Step_time", end_time - step_time, prov4ml.Context.TRAINING, step=e)

            grad_norm = compute_gradient_norm_per_layer(model)
            for k, v in grad_norm.items(): 
                prov4ml.log_metric(f"Grad_norm_{k}", v, prov4ml.Context.TRAINING, step=e)

            prov4ml.log_metric("Indices", indices.tolist(), prov4ml.Context.TRAINING, step=e)
            prov4ml.log_metric("Loss", loss.item(), prov4ml.Context.TRAINING, step=e)
            prov4ml.log_metric("Outlier", is_outlier.tolist(), prov4ml.Context.TRAINING, step=e)

            # prov4ml.log_system_metrics(context=prov4ml.Context.TRAINING, step=e)
            

def test_model(configs : RunConfig, model, criterion, test_loader):
    model.eval()
    acc_criterion = 0
    for _, _, X_batch, y_batch in tqdm(test_loader):
        X_batch, y_batch = X_batch.to(configs.run.device), y_batch.to(configs.run.device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        acc_criterion += loss.item()

    return acc_criterion / len(test_loader)


def main(configs : RunConfig): 

    os.environ['HF_HOME'] = CACHE_PATH
    os.environ['HF_DATASETS_CACHE'] = CACHE_PATH
    os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    LOAD = False

    pynvml.nvmlInit()

    if configs.model.type == "granite": 
        torch.set_default_dtype(torch.bfloat16)


    prov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name="IBM_outliers", 
        provenance_save_dir=f"{CACHE_PATH}/prov",
        save_after_n_logs=100,
        collect_all_processes=False, 
        disable_codecarbon = True, 
    )

    prov4ml.log_artifact("./IBM/config.yaml", prov4ml.Context.TRAINING)

    criterion = get_criterion(configs)
    if not LOAD: 
        dataset = get_dataset(configs, split="train")
        train_loader = get_dataloader(configs, dataset)
        model = get_model(configs).to(configs.run.device)
        optimizer = get_optimizer(configs, model)

        train_model(configs, model, criterion, optimizer, train_loader)
        # torch.save(model, f"final_model_shuffle_{configs.dataset.shuffle}.pth")

    # configs.dataset.batch_size = 8
    # model = torch.load(f"final_model_shuffle_{configs.dataset.shuffle}.pth", weights_only=False)
    # dataset = get_dataset(configs, split="test")
    # test_loader = get_dataloader(configs, dataset)
    # stat = test_model(configs, model, criterion, test_loader)
    # print(stat)
    # prov4ml.log_param("test_loss", stat)

    prov4ml.end_run()

    calculate_metrics(configs)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf')  
    args = parser.parse_args()

    configs = RunConfig(args.conf)
    main(configs)