
import random
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH
from local_datasets.common import *


class ChatDoctorDataset(Dataset):
    def __init__(self, configs : RunConfig, split="train"):
        super().__init__()

        self.context_len = configs.dataset.context_len
        data = load_dataset("avaliev/chat_doctor", cache_dir=CACHE_PATH)[split]
        
        self.inputs = data["input"][:configs.dataset.samples_per_class * configs.dataset.classes]
        self.outputs = data["output"][:configs.dataset.samples_per_class * configs.dataset.classes]
        self.indices = list(range(len(self.inputs)))
        random.shuffle(self.indices)


        self.tokenizer = AutoTokenizer.from_pretrained(configs.model.model_path, token="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")
        if tokenizer_requires_pad_token(configs): 
            self.tokenizer.pad_token = '[PAD]'
            self.tokenizer.pad_token_id = 0

        self.inputs = inject_text_outliers(self.inputs, configs.dataset.outlier_number, self.context_len)
        self.outputs = inject_text_outliers(self.outputs, configs.dataset.outlier_number, self.context_len)
        self.is_outlier = get_is_outlier_list(len(self.inputs), configs.dataset.outlier_number)
    

    def __len__(self): 
        return len(self.inputs)

    def __getitem__(self, index):
        index = self.indices[index]
        inp, out = self.inputs[index], self.outputs[index]  

        input_tokens = self.tokenizer(inp, return_tensors="pt", padding="max_length", truncation=True, max_length=self.context_len)
        output_tokens = self.tokenizer(out, return_tensors="pt", padding="max_length", truncation=True, max_length=self.context_len)["input_ids"].squeeze().float()
        
        return index, self.is_outlier[index], input_tokens, output_tokens