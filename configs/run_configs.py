import yaml

def get_attr_or(o, a, e):
    if a not in o.keys(): 
        print(f"Attribute {a} not found, defaulting to {e}")
        return e
    return o[a]

class ModelConfigs(): 
    def __init__(self, d):
        self.type = get_attr_or(d["model"], "type","mlp")
        self.inner_size = get_attr_or(d["model"], "inner_size", 1280)
        self.model_path = get_attr_or(d["model"], "model_path", None)
        self.perturbate_embeddings = get_attr_or(d["model"], "perturbate_embeddings", False)

class DatasetConfigs(): 
    def __init__(self, d):
        self.type = get_attr_or(d["dataset"], "type", "samples")
        self.samples_per_class = get_attr_or(d["dataset"], "samples_per_class", 20000)
        self.classes = get_attr_or(d["dataset"], "classes", 2)
        self.shuffle = get_attr_or(d["dataset"], "shuffle", True) 
        self.outlier_number = get_attr_or(d["dataset"], "outlier_number", 10)
        self.outlier_range = get_attr_or(d["dataset"], "outlier_range", 0.05)
        self.outlier_distance = get_attr_or(d["dataset"], "outlier_distance", 0.01) 
        self.batch_size = get_attr_or(d["dataset"], "batch_size", 32)
        self.context_len = get_attr_or(d["dataset"], "context_len", 500)

class RunHelpersConfig(): 
    def __init__(self, d):
        self.optimizer = get_attr_or(d["run"], "optimizer", "Adam")
        self.criterion = get_attr_or(d["run"], "criterion", "CrossEntropyLoss")
        self.epochs = get_attr_or(d["run"], "epochs", 3)
        self.learning_rate = get_attr_or(d["run"], "learning_rate", 1e-3)
        self.device = get_attr_or(d["run"], "device", "cuda")

class DetectionConfig(): 
    def __init__(self, d):
        self.keep_epochs = get_attr_or(d["detection"], "keep_epochs", [])
        self.use_pca = get_attr_or(d["detection"], "use_pca", False)
        self.techniques = get_attr_or(d["detection"], "techniques", [])
        self.window_size = get_attr_or(d["detection"], "window_size", 5)
        self.metrics = get_attr_or(d["detection"], "metrics", ["Loss"])
        
class IdentificationConfig(): 
    def __init__(self, d):
        self.window_size = get_attr_or(d["identification"], "window_size", 5)
        self.use_pca = get_attr_or(d["identification"], "use_pca", False)
        self.keep_epochs = get_attr_or(d["identification"], "keep_epochs", [])
        self.techniques = get_attr_or(d["identification"], "techniques", [])
        self.metrics = get_attr_or(d["identification"], "metrics", ["Loss"])

class RunConfig(): 
    def __init__(self, path):

        with open(path, 'r') as file: 
            d = yaml.safe_load(file)

        self.model = ModelConfigs(d)
        self.dataset = DatasetConfigs(d)
        self.run = RunHelpersConfig(d)
        self.detection = DetectionConfig(d)
        self.identification = IdentificationConfig(d)