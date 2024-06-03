import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample

    def forward(self, x):
        out = self.module(x)
        if self.downsample is not None:
            out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet9(nn.Module):
    def __init__(self, cfg):
        super(ResNet9, self).__init__()
        self.base_width = 32
        self.cfg = cfg
        self.in_channels = cfg.task.in_channels
        self.num_class = cfg.task.num_class
        block = BasicBlock

        # input layer
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.base_width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.base_width),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # intermediate layers
        self.layer1 = self._make_layer(block, 32, 1)
        self.layer2 = self._make_layer(block, 64, 1)
        self.layer3 = self._make_layer(block, 128, 1, stride=2)
        self.layer4 = self._make_layer(block, 256, 1, stride=2)

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=256 * block.expansion, out_features=128, bias=False),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=self.num_class, bias=False),
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

    def _make_layer(self, block, planes, num_layers, stride=1):
        # define downsample operation
        downsample = None
        if stride != 1 or self.base_width != planes * block.expansion:
            dconv = nn.Conv2d(in_channels=self.base_width, out_channels=planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            dbn = nn.BatchNorm2d(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        # construct layers
        layers = []
        layers.append(block(self.base_width, planes, stride, downsample))
        self.base_width = planes * block.expansion
        for i in range(1, num_layers):
            layers.append(block(self.base_width, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.maxpool(x)

        # intermediate layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # classifier
        pred = self.classifier(x)

        return pred
