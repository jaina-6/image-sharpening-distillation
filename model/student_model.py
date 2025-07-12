import torch.nn as nn

class LightSharpenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),  # Depthwise
            nn.ReLU(),
            nn.Conv2d(16, 3, 1)  # Pointwise
        )

    def forward(self, x):
        return self.net(x)
