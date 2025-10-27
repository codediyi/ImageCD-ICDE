import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBlock(nn.Module):
    def __init__(self, dim, rates,bm=0.1):
        super(MultiBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate), 
                    nn.Conv2d(dim, dim // len(rates),3, padding=0, dilation=rate),
                    nn.BatchNorm2d( dim // len(rates),bm), 
                    nn.LeakyReLU(0.2, inplace=True),
                    
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        return out
