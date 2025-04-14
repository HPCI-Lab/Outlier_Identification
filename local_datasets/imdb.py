
import random
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH
from local_datasets.common import *


class ImdbReviewsDataset(Dataset):
    def __init__(self, configs : RunConfig, split="train"):
        super().__init__()

        self.num_classes = configs.dataset.classes
        self.context_len = configs.dataset.context_len
        self.data = load_dataset("imdb", cache_dir=CACHE_PATH)[split]
        
        self.texts = self.data["text"][:configs.dataset.samples_per_class*self.num_classes]
        self.labels = self.data["label"][:configs.dataset.samples_per_class*self.num_classes]
        self.texts = inject_text_outliers(self.texts, configs.dataset.outlier_number, self.context_len)
        self.labels = inject_label_outliers(self.labels, configs.dataset.outlier_number, self.num_classes)
        self.is_outlier = get_is_outlier_list(len(self.texts), configs.dataset.outlier_number)
        
        self.indices = list(range(configs.dataset.samples_per_class * configs.dataset.classes))
        random.shuffle(self.indices)

        self.tokenizer = AutoTokenizer.from_pretrained(configs.model.model_path, token="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")
        if tokenizer_requires_pad_token(configs): 
            self.tokenizer.pad_token = '[PAD]'
            

    def __len__(self): 
        return len(self.texts)

    def __getitem__(self, index):
        index = self.indices[index]
        inp, out = self.texts[index], self.labels[index]
        input_tokens = self.tokenizer(inp, return_tensors="pt", padding="max_length", truncation=True, max_length=self.context_len)

        if self.is_outlier[index]:
            for i in range(len(input_tokens["input_ids"])):
                # input_tokens["input_ids"][i] = random.randint(vocab_size - 1000, vocab_size - 1)
                input_tokens["input_ids"] = torch.flip(input_tokens["input_ids"], dims=[0])

        return index, self.is_outlier[index], input_tokens, torch.tensor(out)

