
from transformers import AutoModelForCausalLM
from torch import nn
from configs.run_configs import RunConfig
from configs.paths import CACHE_PATH

class LlamaFinetunePipeline(nn.Module): 

    def __init__(self, configs : RunConfig):
        super().__init__()
        
        self.device = configs.run.device
        self.context_len = configs.dataset.context_len
        self.model = AutoModelForCausalLM.from_pretrained(configs.model.model_path, device_map=self.device, token="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", cache_dir=CACHE_PATH)

        if configs.dataset.type in ["imdb"]: 
            self.model.lm_head = nn.Linear(2048, 3).to(self.device)

        # from peft import get_peft_model, LoraConfig
        # lora_config = LoraConfig(
        #     r=8,
        #     target_modules="all-linear", #get_specific_layer_names(self.model), 
        #     lora_alpha=16,  # lora_alpha
        #     lora_dropout=0.1,  # Dropout for LoRA
        #     bias="none",  # No bias
        #     task_type="SEQ_2_SEQ_LM"  # Sequence Classification
        # )
        # self.model = get_peft_model(self.model, lora_config)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.lm_head.requires_grad_(True)


    def forward(self, x, y): 
        x = self.model(input_ids=x["input_ids"].squeeze(1), attention_mask=x["attention_mask"].squeeze(1), labels=y)
        l = x["loss"]
        x = x["logits"].mean(dim=-1)
        return x, l