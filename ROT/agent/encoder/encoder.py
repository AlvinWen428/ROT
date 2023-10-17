import torch.nn as nn
from agent.encoder.backbone import build_backbone
import utils

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Encoder(nn.Module):
    def __init__(self, backbone_name, obs_shape, pretrained, device):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_frames = obs_shape[0] // 3
        self.img_size = obs_shape[-2:]

        self.backbone = build_backbone(backbone_name, pretrained, self.img_size, self.num_frames)
        self.repr_dim = self.backbone.num_features

        self.img_normalizer = utils.ImageNormalize(max_value=255, mean=IMAGENET_MEAN, std=IMAGENET_STD, device=device)

    def forward(self, obs):
        """
        obs: [T, L*C, H, W]
        """
        T, LC, H, W = obs.shape
        assert LC == (self.num_frames * 3)
        obs = obs.view(T, self.num_frames, 3, H, W)

        obs = self.img_normalizer(obs)

        h = self.backbone(obs)
        h = h.view(h.shape[0], -1)
        return h
