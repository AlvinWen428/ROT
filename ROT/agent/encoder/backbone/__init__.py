from agent.encoder.backbone.conv import SimpleConv
import agent.encoder.backbone.video_mae as video_mae
from utils import weight_init


def build_backbone(backbone_name, pretrained, img_size, num_frames):
    if backbone_name == "simple_conv":
        backbone = SimpleConv(num_frames * 3, img_size)
        backbone.apply(weight_init)
    elif "videomae" in backbone_name:
        kwargs = dict(pretrained=pretrained,
                      use_global_pool=True,
                      img_size=img_size,
                      num_frames=num_frames)
        backbone = getattr(video_mae, backbone_name)(**kwargs)
    else:
        raise ValueError
    return backbone
