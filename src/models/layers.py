import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Input : (B, T, D)
    Output: (B, T, D)
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):

        rms = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True)
            + self.eps
        )

        x = x * rms

        return self.weight * x



class DepthwiseConv1D(nn.Module):
    """
    Depthwise temporal convolution.

    Input :
        (B,T,D)

    Output :
        (B,T,D)
    """

    def __init__(
        self,
        dim,
        kernel_size=4
    ):

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=dim
        )

    def forward(self, x):

        x = x.transpose(1,2)   # coz (B,D,T) is expected by Conv1D

        x = self.conv(x)

        x = x[:,:,:-3]         # remove extra padding

        x = x.transpose(1,2)

        return x



class SiLUGate(nn.Module):

    def forward(self, x, gate):

        return x * F.silu(gate)