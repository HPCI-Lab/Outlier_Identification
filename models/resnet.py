
from torch import nn
import torchvision

class ResnetWrapper(nn.Module): 
    def __init__(self, in_channels, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model = torchvision.models.resnet18()
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        model.fc = nn.Linear(512 * torchvision.models.resnet.BasicBlock.expansion, num_classes)
        self.model = model#.to(configs.run.device)

    def forward(self, x, y=None): 
        x = self.model(x)
        return x, None