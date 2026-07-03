import torch
import sys 
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from models.mamba import MambaBlock

x = torch.randn(
    2,
    16,
    512
)

model = MambaBlock()

y = model(x)

print(y.shape)