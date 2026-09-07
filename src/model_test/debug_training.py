import numpy as np
from src.datasets import SweepsDataset


dataset = SweepsDataset(
    "tus-rec",
    split="train",
    mode="h5_dynamic_load",
    original_image_shape=(480, 640),
)

print("Number of scans:", len(dataset))

for idx in range(len(dataset)):
    item = dataset[idx]

    tracking = np.asarray(item["tracking"])

    rotations = tracking[:, :3, :3]

    for frame_idx, R in enumerate(rotations):
        det = np.linalg.det(R)

        if det <= 0:
            print("\nBAD ROTATION FOUND")
            print("dataset index:", idx)
            print("sweep_id:", item["sweep_id"])
            print("frame:", frame_idx)
            print("det:", det)
            print("R:")
            print(R)
            print("tracking matrix:")
            print(tracking[frame_idx])

            raise SystemExit

print("\nAll rotations passed determinant test.")