import torch

ckpt = torch.load(
    "/home/user/Desktop/ULTRASOUND/DualtrackPrediction/experiments/local_stage2/checkpoint/best.pt",
    map_location="cpu",
    weights_only=False
)

print("Epoch:", ckpt["epoch"])
print("Best score:", ckpt["best_score"])