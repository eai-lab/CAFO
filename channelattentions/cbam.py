"""sourced from https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py"""
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import weight_norm


class CBAMAttention(nn.Module):
    """Channel Attention from CBAM"""

    def __init__(self, cfg):
        super(CBAMAttention, self).__init__()
        self.cfg = cfg
        assert self.cfg.channelattention.name == "cbam"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        in_planes = cfg.task.in_channels
        hid_planes = cfg.channelattention.reduction_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_planes, out_features=hid_planes, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hid_planes, out_features=in_planes, bias=True),
        )

        self.sigmoid = nn.Sigmoid()

    #     self.fc.apply(self.init_weights)

    # def init_weights(self, m):
    #     if type(m) == nn.Linear:
    #         print("initializing weight")
    #         init.uniform_(m.weight, -0.1, 0.1)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = self.fc(avg_out)

        max_out_original = self.max_pool(x)
        max_out = max_out_original.view(max_out_original.size(0), -1)
        max_out = self.fc(max_out)
        out = avg_out + max_out
        channel_attention = self.sigmoid(out)
        channel_attention = channel_attention.view(channel_attention.size(0), channel_attention.size(1), 1, 1)
        return channel_attention
