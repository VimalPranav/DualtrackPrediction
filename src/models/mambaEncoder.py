import torch.nn as nn

from src.models.mamba import MambaBlock


class MambaEncoder(nn.Module):
    """
    Stack of Mamba Blocks.

    Input:
        (B,T,D)

    Output:
        (B,T,D)
    """

    def __init__(
        self,
        d_model=512,
        num_layers=2,
        expand=2,
        d_state=64,
        kernel_size=4,
        dropout=0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                expand=expand,
                d_state=d_state,
                kernel_size=kernel_size
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        for layer in self.layers:
            
            x = layer(x)
            x = self.dropout(x)

        return x