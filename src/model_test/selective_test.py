import torch
import sys 
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from models.selective_scan import SelectiveScan

x = torch.randn(
    2,
    16,
    512
)

model = SelectiveScan()

y = model(x)

print(y.shape)