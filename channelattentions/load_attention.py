
from .depthwise import DepthWiseAttention
from .cbam import CBAMAttention



def load_channel_attention_module(cfg):
    if cfg.channelattention.name == "cbam":
        print("Loading CBAM Attention")
        return CBAMAttention(cfg)
    elif cfg.channelattention.name == "depthwise":
        print("Loading DepthWise Attention")
        return DepthWiseAttention(cfg)
    else:
        raise NotImplementedError
