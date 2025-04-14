
from torch import nn
import torchvision

class SwinWrapper(nn.Module): 
    def __init__(self, size, num_classes):
        super().__init__()

        if size == "tiny": 
            self.model = torchvision.models.swin_t(num_classes=num_classes) 
        elif size == "small": 
            self.model = torchvision.models.swin_s(num_classes=num_classes) 
        elif size == "big": 
            self.model = torchvision.models.swin_b(num_classes=num_classes) 
        elif size == "big2": 
            self.model = torchvision.models.swin_v2_b(num_classes=num_classes) 

        # pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        # print(pytorch_total_params)

    def forward(self, x, y=None): 
        x = self.model(x)
        return x, None