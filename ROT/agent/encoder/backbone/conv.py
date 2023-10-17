import torch.nn as nn


class SimpleConv(nn.Module):
    def __init__(self, in_channel, img_size):
        super(SimpleConv, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channel, 32, 3, stride=2), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())
        h, w = img_size
        out_h = int(((h - 3) / 2 + 1) - 2 - 2 - 2)
        out_w = int(((h - 3) / 2 + 1) - 2 - 2 - 2)
        self.num_features = 32 * out_h * out_w

    def forward(self, x):
        """
        x: [T, L, C, H, W]
        """
        T, L, C, H, W = x.shape
        x = x.view(T, L*C, H, W)
        return self.net(x)
