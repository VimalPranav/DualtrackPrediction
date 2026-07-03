import torch
import sys 
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from models.mambaEncoder import MambaEncoder

x = torch.randn(
    2,
    16,
    512
)

encoder = MambaEncoder(
    num_layers=2
)

y = encoder(x)

print(y.shape)