import torch
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)


from models.layers import (
    RMSNorm,
    DepthwiseConv1D,
    SiLUGate
)

x = torch.randn(
    2,
    16,
    512
)

norm = RMSNorm(512)

conv = DepthwiseConv1D(512)

gate = SiLUGate()

y = norm(x)

print(y.shape)

y = conv(y)

print(y.shape)

z = torch.randn_like(y)

y = gate(y,z)

print(y.shape)