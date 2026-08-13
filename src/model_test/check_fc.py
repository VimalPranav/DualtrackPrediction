import torch
from src.models import get_model   # adjust if your import is different

ckpt_path = "/home/user/Desktop/ULTRASOUND/DualtrackPrediction/experiments/local_stage2/checkpoint/best.pt"

stage2 = get_model(
    name="dualtrack_loc_enc_stg2",
    checkpoint=ckpt_path
)

print("Stage2 FC weight norm:",
      stage2.fc.weight.norm().item())

print("Stage2 FC bias norm:",
      stage2.fc.bias.norm().item())