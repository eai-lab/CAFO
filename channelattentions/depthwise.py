from torch import nn

## DepCA Implementation.
class DepthWiseAttention(nn.Module):
    def __init__(self, cfg):
        super(DepthWiseAttention, self).__init__()
        self.cfg = cfg
        assert self.cfg.channelattention.name == "depthwise"
        in_planes = cfg.task.in_channels
        expansion_filter_num = cfg.channelattention.expansion_filter_num
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_planes,
            out_channels=in_planes * expansion_filter_num,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=in_planes,
            bias=True,
        )

        self.max_filter_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_filter_pool = nn.AdaptiveAvgPool2d(1)

        self.max_1d_pool = nn.AvgPool1d(
            kernel_size=expansion_filter_num,
            stride=expansion_filter_num,
        )
        self.avg_1d_pool = nn.AvgPool1d(
            kernel_size=expansion_filter_num,
            stride=expansion_filter_num,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_model_outputs=False):
        depthwise_conv_out = self.depthwise_conv(x)
        max_filter_pool_out = self.max_filter_pool(depthwise_conv_out)
        avg_filter_pool_out = self.avg_filter_pool(depthwise_conv_out)

        avg_1d_pool_out = self.avg_1d_pool(avg_filter_pool_out.squeeze())
        max_1d_pool_out = self.max_1d_pool(max_filter_pool_out.squeeze())

        out = avg_1d_pool_out + max_1d_pool_out

        channel_attention = self.sigmoid(out)
        channel_attention = channel_attention.view(channel_attention.size(0), channel_attention.size(1), 1, 1)
        if return_model_outputs:
            return channel_attention, depthwise_conv_out
        else:
            return channel_attention
