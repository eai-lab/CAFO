import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import SimpleViT
from torch.nn import init


class SimpleVIT(nn.Module):
    def __init__(self, cfg):
        super(SimpleVIT, self).__init__()
        self.cfg = cfg
        self.model_params = cfg["model"]
        self.in_channels = cfg.task.in_channels
        image_size = self.cfg.task.vit_params.image_size
        patch_size = self.cfg.task.vit_params.patch_size
        num_classes = cfg.task.num_class
        dim = self.model_params.dim
        depth = self.model_params.depth
        heads = self.model_params.heads
        mlp_dim = self.model_params.mlp_dim


        self.VIT = SimpleViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            channels = self.in_channels,
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self,x):
        pred = self.VIT(x)
        return pred