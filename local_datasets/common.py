from configs.run_configs import RunConfig
import random, string

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def tokenizer_requires_pad_token(configs : RunConfig): 
    return configs.model.type in ["mistral", "llama", "mixtral"]

def inject_label_outliers(labels, outlier_number, num_classes): 
   for i in range(0, outlier_number): 
      lbls = list(range(num_classes))
      lbls.remove(labels[i])
      labels[i] = random.choice(lbls)
   return labels

def inject_text_outliers(texts, outlier_number, context_len): 
   for i in range(0, outlier_number): 
      texts[i] = " ".join([randomword(20) for _ in range(context_len)])
   return texts

def inject_img_outliers(imgs, outlier_number): 
   for i in range(0, outlier_number): 
      imgs[i] = 255 - imgs[i]
   return imgs

def get_is_outlier_list(total_samples, outlier_number): 
   return  [i < outlier_number for i in range(total_samples)]
