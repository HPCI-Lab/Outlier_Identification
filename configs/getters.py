
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader

from local_datasets.mnist import OutlierMNISTDatasetWrapper
from local_datasets.samples import OutlierSamplesDataset
from local_datasets.cifar import CIFARDatasetWrapper
from local_datasets.imdb import ImdbReviewsDataset
from local_datasets.med_magic import HealthCareMagicDataset, HealthCareMagicChatDataset
from local_datasets.chat_doctor import ChatDoctorDataset
from models.mlp import MLP
from models.cnn import CNN
from models.swin import SwinWrapper
from models.bert import BertFinetunePipeline
from models.albert import AlbertFinetunePipeline
from models.granite import GraniteFinetunePipeline
from models.llama import LlamaFinetunePipeline
from models.mistral import MistralFinetunePipeline
from models.resnet import ResnetWrapper
from models.lstm import LSTMWrapper
import prov4ml
import subprocess
import GPUtil
import re

def get_model(configs : RunConfig): 
    if configs.model.type == "mlp": 

        if configs.dataset.type == "samples": 
            in_channels = 2
        elif configs.dataset.type == "mnist": 
            in_channels = 28*28

        return MLP(in_channels, configs.model.inner_size, configs.dataset.classes)#.to(configs.run.device)
    
    elif configs.model.type == "resnet18": 
        in_channels = 1 if configs.dataset.type == "mnist" else 3
        return ResnetWrapper(in_channels, configs.dataset.classes)

    elif configs.model.type == "cnn":
        return CNN(1, 256, configs.dataset.classes)#.to(configs.run.device)

    elif configs.model.type == "swin": 
        return SwinWrapper(configs.model.model_path, configs.dataset.classes)
    
    elif configs.model.type == "bert": 
        return BertFinetunePipeline(configs)

    elif configs.model.type == "albert": 
        return AlbertFinetunePipeline(configs)

    elif configs.model.type == "tinyvit": 
        from models.tiny_vit import tiny_vit_21m_224
        return tiny_vit_21m_224(pretrained=True, num_classes=configs.dataset.classes)

    elif configs.model.type in ["lstm_small", "lstm_medium", "lstm_large"]: 
        s2s = configs.dataset.type != "imdb"
        cls = configs.dataset.classes if not s2s else configs.dataset.context_len
        return LSTMWrapper(configs.model.type, cls, seq2seq=s2s)

    elif configs.model.type == "granite": 
        return GraniteFinetunePipeline(configs)

    elif configs.model.type == "llama": 
        return LlamaFinetunePipeline(configs)
    
    elif configs.model.type == "mistral" or configs.model.type == "mixtral": 
        return MistralFinetunePipeline(configs)

    else: 
        raise AttributeError("Model type not found")


def get_criterion(configs : RunConfig): 
    if configs.run.criterion == "CrossEntropyLoss": 
        return nn.CrossEntropyLoss()
    elif configs.run.criterion == "MSELoss": 
        return nn.MSELoss()
    else: 
        raise AttributeError("Critetion type not found")


def get_gpu_power_usage():
    gpu_id = GPUtil.getGPUs()[0].id
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
        ).decode().strip()
        power_usage = float(re.findall(r'\d+\.\d+', output)[0])
        return power_usage
    except Exception as e:
        print("Error fetching GPU power usage:", e)
        return 0.0


def get_optimizer(configs : RunConfig, model): 
    if configs.run.optimizer == "adam": 
        return optim.Adam(model.parameters(), lr=configs.run.learning_rate)
    elif configs.run.optimizer == "fused_adam": 
        import deepspeed
        return deepspeed.ops.adam.FusedAdam(model.parameters(), lr=configs.run.learning_rate)
    elif configs.run.optimizer == "ds_cpu_adam": 
        import deepspeed
        return deepspeed.ops.lamb.FusedLamb(model.parameters(), lr=configs.run.learning_rate)
    else: 
        raise AttributeError("Optimizer type not found")


def get_dataset(configs: RunConfig, split="train"): 
    if configs.dataset.type == "samples": 
        dataset = OutlierSamplesDataset(
            configs.dataset.samples_per_class, 
            configs.dataset.classes, 
            outlier_dist=configs.dataset.outlier_distance, 
            outlier_range=configs.dataset.outlier_range, 
            outlier_perc=configs.dataset.outlier_percentage)

    elif configs.dataset.type == "mnist": 

        if configs.model.type == "tinyvit": 
            tform = transforms.Compose([transforms.Resize((224,224)), transforms.PILToTensor()])
        else: 
            tform = transforms.Compose([transforms.PILToTensor()])

        dataset = torchvision.datasets.MNIST(root=CACHE_PATH, download=True, train=True, transform=tform)
        dataset = OutlierMNISTDatasetWrapper(
            dataset, 
            outlier_num=configs.dataset.outlier_number, 
            samples_per_class=configs.dataset.samples_per_class, 
            classes=configs.dataset.classes, 
        )

    elif configs.dataset.type == "cifar100": 

        if configs.model.type == "tinyvit": 
            tform = transforms.Compose([transforms.Resize((224,224)), transforms.PILToTensor()])
        else: 
            tform = transforms.Compose([transforms.PILToTensor()])

        dataset = torchvision.datasets.CIFAR100(root=CACHE_PATH, download=True, train=True, transform=tform)
        dataset = CIFARDatasetWrapper(
            dataset, 
            outlier_num=configs.dataset.outlier_number, 
            samples_per_class=configs.dataset.samples_per_class, 
            classes=configs.dataset.classes, 
        )

    elif configs.dataset.type == "med":
        dataset = HealthCareMagicDataset(configs, split=split)

    elif configs.dataset.type == "chat_med":
        dataset = HealthCareMagicChatDataset(configs, split=split)

    elif configs.dataset.type == "chat_doctor":
        dataset = ChatDoctorDataset(configs, split=split)

    elif configs.dataset.type == "imdb": 
        dataset = ImdbReviewsDataset(configs, split=split)

    else: 
        raise AttributeError("Dataset type not found")

    for _, o, _, _ in dataset: 
        prov4ml.log_metric("Outliers", 1.0 if o else 0.0, prov4ml.Context.TRAINING, step=0)
    return dataset


def get_dataloader(configs: RunConfig, dataset): 
    return DataLoader(dataset, batch_size=configs.dataset.batch_size, shuffle=configs.dataset.shuffle)
