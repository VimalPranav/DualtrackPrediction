import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveScan(nn.Module):
    """
    Custom Input-Selective State Space Layer

    Input:
        (B, T, D)

    Output:
        (B, T, D)
    """

    def __init__(
        self,
        d_model=512,
        d_state=64
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state

        # Input-dependent gates
        self.delta_proj = nn.Linear(d_model, d_state)       # How quickly the memory should change?
        self.B_proj = nn.Linear(d_model, d_state)           # How much current frame enters memory?
        self.C_proj = nn.Linear(d_model, d_state)           # How much memory becomes output?

        # Hidden state transition
        self.A = nn.Parameter(
            torch.randn(d_state) * 0.02
        )

        # Output projection
        self.output_proj = nn.Linear(
            d_state,
            d_model
        )

    def forward(self, x):

        """
        x : (B,T,D)
        """

        B, T, D = x.shape

        device = x.device

        h = torch.zeros(
            B,
            self.d_state,
            device=device
        )

        outputs = []

        for t in range(T):

            xt = x[:, t]

            # Input-dependent parameters

            delta = F.softplus(                # ensures positive answer
                self.delta_proj(xt)
            )

            B_t = torch.tanh(                  # range [-1,1]
                self.B_proj(xt)
            )

            C_t = torch.sigmoid(               # range [0,1]
                self.C_proj(xt)
            )

            # State update

            A = torch.exp(
                -delta * self.A
            )

            h = (
                A * h
                +
                B_t
            )

            # Output

            y = C_t * h

            y = self.output_proj(
                y
            )

            outputs.append(y)

        outputs = torch.stack(
            outputs,
            dim=1
        )

        return outputs