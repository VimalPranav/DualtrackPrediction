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
        self.A_log = nn.Parameter(
            torch.zeros(d_state)
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
            device=device,
            dtype=x.dtype
        )

        outputs = torch.empty(
            B,
            T,
            D,
            device=device,
            dtype=x.dtype
        )
        delta_all = F.softplus(
            self.delta_proj(x)
        )

        B_all = torch.tanh(
            self.B_proj(x)
        )

        C_all = torch.sigmoid(
            self.C_proj(x)
        )

        A_base = -torch.exp(self.A_log)
        

        for t in range(T):

            # Input-dependent parameters

            delta = delta_all[:, t]
            B_t = B_all[:, t]
            C_t = C_all[:, t]

            # State update

            A = torch.exp(
                delta * A_base
            )

            h = (
                A * h
                +
                B_t
            )

            if not torch.isfinite(h).all():
                raise RuntimeError(
                    f"Hidden state contains NaN/Inf at timestep {t}"
                )

            # Output

            y = C_t * h

            y = self.output_proj(
                y
            )

            outputs[:, t] = y


        return outputs