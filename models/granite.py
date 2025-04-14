
import torch

from transformers import AutoModelForCausalLM
from torch import nn
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH

class GraniteFinetunePipeline(nn.Module): 

    def __init__(self, configs : RunConfig):
        super().__init__()
        
        self.device = configs.run.device
        self.context_len = configs.dataset.context_len
        self.model = AutoModelForCausalLM.from_pretrained(configs.model.model_path, device_map=self.device, cache_dir=CACHE_PATH)
        self.task = "seq2seq"
        self.perturbate_embeddings = configs.model.perturbate_embeddings

        if configs.dataset.type in ["imdb"]: 
            self.task = "cls"
            self.model.lm_head = nn.Linear(2048, configs.dataset.classes).to(self.device)
        # else: 
        #     self.model.lm_head = nn.Linear(2048, configs.dataset.context_len).to(self.device)

        # from peft import get_peft_model, LoraConfig, TaskType
        # lora_config = LoraConfig(
        #     r=8,
        #     target_modules="all-linear", #get_specific_layer_names(self.model), 
        #     lora_alpha=16,  # lora_alpha
        #     lora_dropout=0.1,  # Dropout for LoRA
        #     bias="none",  # No bias
        #     task_type=TaskType.SEQ_2_SEQ_LM  # Sequence Classification
        # )
        # self.model = get_peft_model(self.model, lora_config)

        layers = list(self.model.children())
        for layer in layers[:-2]:  # Skip the last layers
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x, y=None): 
        # if self.perturbate_embeddings: 
        #     with torch.no_grad():
        #         original_embeddings = self.model.embed_tokens.word_embeddings(x["input_ids"].squeeze(1))
        #     random_embeddings = torch.randn_like(original_embeddings)
        #     x = self.model(inputs_embeds=random_embeddings, attention_mask=x["attention_mask"].squeeze(1))
        #     x = x["logits"].mean(dim=-1)
        #     return x, l
        # else: 
        # if self.task == "cls": 
        x = self.model(input_ids=x["input_ids"].squeeze(1), attention_mask=x["attention_mask"].squeeze(1))
        # else: 
        #     x = self.model(input_ids=x["input_ids"].squeeze(1), attention_mask=x["attention_mask"].squeeze(1), labels=y)
        #     l = x["loss"]
        x = x["logits"].mean(dim=-1)
        return x, None