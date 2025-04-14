
import argparse
import prov4ml
import time
import torch 
import lightning as L
import os
import torch.nn.functional as F

from utils.metrics import calculate_metrics
from configs.getters import * 
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH

def main(configs : RunConfig): 

    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "50000"
    os.environ["MAX_JOBS"] = "4"
    os.environ['HF_HOME'] = CACHE_PATH
    os.environ['HF_DATASETS_CACHE'] = CACHE_PATH
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    # os.environ['TORCH_CUDA_ARCH_LIST'] = "7.0 7.5 8.0"

    prov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name="IBM_outliers", 
        provenance_save_dir=f"{CACHE_PATH}/prov",
        save_after_n_logs=100,
        collect_all_processes=False, 
        disable_codecarbon=True, 
    )

    # prov4ml.log_artifact("./IBM/config.yaml", prov4ml.Context.TRAINING)

    dataset = get_dataset(configs)
    train_loader = get_dataloader(configs, dataset)

    m = LitAutoEncoder()
    trainer = L.Trainer(
        max_epochs=configs.run.epochs, 
        accelerator="gpu",
        strategy="deepspeed_stage_3", 
        # strategy="fsdp", 
        precision="bf16-true", 
        # plugins=precision,
        default_root_dir=CACHE_PATH, 
        devices=1,
        limit_val_batches=0, 
        # limit_train_batches=1, 
        limit_test_batches=0, 
        logger=[], 
    )
    trainer.fit(model=m, train_dataloaders=train_loader)
    
    prov4ml.end_run()

    # calculate_metrics(configs)


class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_model(configs)
        self.criterion = get_criterion(configs)

    def training_step(self, batch, batch_idx):
        step_time = time.time()

        indices, is_outlier, X_batch, y_batch = batch

        if configs.dataset.type in ["mnist"] and configs.model.type in ["tinyvit"]:
            X_batch = torch.cat([X_batch, X_batch, X_batch], dim=1)
        x_hat, loss = self.model(X_batch, None) #y_batch.type(torch.LongTensor))
        if loss is None: 
            if configs.model.type in ["tinyvit"]:
                y_batch = y_batch.float()

            loss = self.criterion(x_hat, y_batch) #/ 30522


        # backward_time = time.time()
        # self.manual_backward(loss)
        # g1, g2 = compute_gradient_norm(self.model.model), compute_gradient_norm_deepspeed(self.model.model)
        # print(g1, g2)

        # optim_time = time.time()
        # optimizer.step()

        end_time = time.time()
        prov4ml.log_metric("Step_time", end_time - step_time, prov4ml.Context.TRAINING, step=self.current_epoch)
        #prov4ml.log_metric("Backward_time", end_time - backward_time, prov4ml.Context.TRAINING, step=self.current_epoch)
        #prov4ml.log_metric("Optim_step_time", end_time - optim_time, prov4ml.Context.TRAINING, step=self.current_epoch)
        prov4ml.log_metric("Indices", indices.tolist(), prov4ml.Context.TRAINING, step=self.current_epoch)
        prov4ml.log_metric("Loss", loss.item(), prov4ml.Context.TRAINING, step=self.current_epoch)
        prov4ml.log_metric("Outlier", is_outlier.tolist(), prov4ml.Context.TRAINING, step=self.current_epoch)
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=self.current_epoch)
        #self.log("Loss", loss.item())

        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(configs, self.model)
        return {
            "optimizer": optimizer,
        }



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf')  
    args = parser.parse_args()

    configs = RunConfig(args.conf)
    main(configs)