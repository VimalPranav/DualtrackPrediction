import torch
import sys 
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from models.mamba_temporal import MambaTracker

x = torch.randn(
    2,
    16,
    1,
    128,
    128
)

model = MambaTracker()

y = model(x)

print(y.shape)