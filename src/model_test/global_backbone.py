import torch
import sys 
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from models.global_encoder import global_encoder_cnn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # --------------------------------------------------
    # Example input
    # B = batch size
    # N = number of ultrasound frames
    # C = image channels
    # H,W = image resolution
    # --------------------------------------------------
    B = 2
    N = 16
    C = 1
    H = 224
    W = 224

    images = torch.randn(B, N, C, H, W).to(device)

    print("\nInput")
    print("images:", images.shape)

    # --------------------------------------------------
    # Build backbone
    # --------------------------------------------------
    model = global_encoder_cnn().to(device)
    model.eval()
    print("features_only =", model.features_only)

    print("\nModel:")
    print(model)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    with torch.no_grad():
        sample_indices = torch.arange(
        images.shape[1],
        device=images.device
    ).unsqueeze(0).expand(images.shape[0], -1)

    features = model(images, sample_indices)
    print("Output type:", type(features))

    if isinstance(features, torch.Tensor):
        print("Output shape:", features.shape)
    else:
        print("Output:", features.keys())

    print("\nOutput")

    print("Output type:", type(features))
    print("Output keys:", features.keys())

    frame_features = features["frame_features"]
    cls_feature = features["cls_feature"]

    print("frame_features:", frame_features.shape)
    print("cls_feature:", cls_feature.shape)

    expected_frame = (B, N, 512)
    expected_cls = (B, 512)

    assert frame_features.shape == expected_frame, (
        f"Expected frame features {expected_frame}, "
        f"got {frame_features.shape}"
    )

    assert cls_feature.shape == expected_cls, (
        f"Expected CLS feature {expected_cls}, "
        f"got {cls_feature.shape}"
    )

    print("✓ Global encoder output test passed")


if __name__ == "__main__":
    main()