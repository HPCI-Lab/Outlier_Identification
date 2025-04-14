
import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH
from local_datasets.common import *

class HealthCareMagicDataset(Dataset):
    def __init__(self, configs : RunConfig, split="train"):
        super().__init__()

        self.context_len = configs.dataset.context_len
        data = load_dataset("wangrongsheng/HealthCareMagic-100k-en", cache_dir=CACHE_PATH)["train"]
        if split == "train": 
            self.inputs = data["input"][:configs.dataset.samples_per_class * configs.dataset.classes]
            self.indices = list(range(configs.dataset.samples_per_class * configs.dataset.classes))
            self.outputs = data["output"][:configs.dataset.samples_per_class * configs.dataset.classes]
        else: 
            self.indices = list(range(configs.dataset.samples_per_class))
            self.inputs = data["input"][-configs.dataset.samples_per_class:]
            self.outputs = data["output"][:configs.dataset.samples_per_class]

        random.shuffle(self.indices)
        self.tokenizer = AutoTokenizer.from_pretrained(configs.model.model_path, token="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")
        if tokenizer_requires_pad_token(configs): 
            self.tokenizer.pad_token = '[PAD]'
            # self.tokenizer.pad_token_id = 0#self.tokenizer.eos_token_id
            # self.tokenizer.pad_token = self.tokenizer.eos_token
            #self.tokenizer.add_eos_token = True

        self.inputs = inject_text_outliers(self.inputs, configs.dataset.outlier_number, self.context_len)
        self.outputs = inject_text_outliers(self.outputs, configs.dataset.outlier_number, self.context_len)
        self.is_outlier = get_is_outlier_list(len(self.inputs), configs.dataset.outlier_number)


    def __len__(self): 
        return len(self.inputs)

    def __getitem__(self, index):
        index = self.indices[index]
        inp, out, is_out = self.inputs[index], self.outputs[index], self.is_outlier[index]
            
        input_tokens = self.tokenizer(inp, return_tensors="pt", padding="max_length", truncation=True, max_length=self.context_len)
        output_tokens = self.tokenizer(out, return_tensors="pt", padding="max_length", truncation=True, max_length=self.context_len)#["input_ids"].squeeze()#.float()

        if is_out: 
            input_tokens["input_ids"] = torch.flip(input_tokens["input_ids"], dims=[0])
            # for i in range(len(input_tokens["input_ids"])): 
            #     input_tokens["input_ids"][i] = random.choice(range(self.tokenizer.vocab_size-100, self.tokenizer.vocab_size))

        if is_out: 
            output_tokens["input_ids"] = torch.flip(output_tokens["input_ids"], dims=[0])
            # for i in range(len(output_tokens)): 
                # output_tokens[i] = random.choice(range(self.tokenizer.vocab_size-100, self.tokenizer.vocab_size))
            # indexes = torch.randomperm(output_tokens["input_ids"].shape)
            # output_tokens["input_ids"] = output_tokens["input_ids"][indexes]


        return index, is_out, input_tokens, output_tokens["input_ids"].squeeze().float()#.type(torch.LongTensor)



class HealthCareMagicChatDataset(Dataset):
    def __init__(self, configs : RunConfig, split="train"):
        super().__init__()

        self.context_len = configs.dataset.context_len
        data = load_dataset("RafaelMPereira/HealthCareMagic-100k-Chat-Format-en", cache_dir=CACHE_PATH)[split]
        sents = data["text"][:configs.dataset.samples_per_class * configs.dataset.classes]
        self.inputs = [sent.split("<bot>")[0].replace("<human>", "") for sent in sents]
        self.outputs = [sent.split("<bot>")[1] for sent in sents]

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