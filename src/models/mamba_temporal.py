import torch
import torch.nn as nn

from models.mambaEncoder import MambaEncoder


class SimpleTemporalMamba(nn.Module):

    def __init__(
        self,
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=32,   
        num_attention_heads=4,  
        features_only=False,
        max_position_embeddings=1024,
        input_size=None,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        self.features_only = features_only

        if input_size is not None and input_size != hidden_size:
            self.proj = nn.Linear(
                input_size,
                hidden_size
            )
        else:
            self.proj = None

        self.encoder = MambaEncoder(
            d_model=hidden_size,
            num_layers=num_hidden_layers,
            dropout=dropout
        )

        self.fc = nn.Sequential(

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, 6)

        )

    def forward(self, features):

        if self.proj is not None:
            features = self.proj(features)

        hidden_state = self.encoder(features)

        if self.features_only:
            return hidden_state

        outputs = self.fc(hidden_state)

        return outputs[:,1:,:]

    def predict(self, data, device):
        return self(data["pooled_cnn_features"].to(device))