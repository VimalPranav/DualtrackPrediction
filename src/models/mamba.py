import torch
import torch.nn as nn

from models.layers import (
    RMSNorm,
    DepthwiseConv1D,
    SiLUGate
)

from models.selective_scan import (
    SelectiveScan
)


class MambaBlock(nn.Module):
    """
    Custom Mamba Block

    Input :
        (B,T,D)

    Output :
        (B,T,D)
    """

    def __init__(
        self,
        d_model=512,
        expand=2,
        d_state=64,
        kernel_size=4
    ):
        super().__init__()

        inner_dim = d_model * expand

        # RMSNorm applied

        self.norm = RMSNorm(d_model)

        # Input Projection

        self.in_proj = nn.Linear(
            d_model,
            inner_dim
        )

        # Temporal Convolution

        self.conv = DepthwiseConv1D(
            inner_dim // 2,
            kernel_size
        )

        # Selective State Space

        self.ssm = SelectiveScan(
            d_model=inner_dim // 2,
            d_state=d_state
        )

        # Gate

        self.gate = SiLUGate()

        # Output Projection

        self.out_proj = nn.Linear(
            inner_dim // 2,
            d_model
        )

    def forward(self, x):

        residual = x

        x = self.norm(x)

        x = self.in_proj(x)

        x_part, gate = torch.chunk(
            x,
            2,
            dim=-1
        )

        x_part = self.conv(
            x_part
        )

        x_part = self.ssm(
            x_part
        )

        x_part = self.gate(
            x_part,
            gate
        )

        x_part = self.out_proj(
            x_part
        )

        return residual + x_part