
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x, l=None):
        B = x.shape[0]
        x = x.reshape(B, -1)
        return self.layers(x), None
