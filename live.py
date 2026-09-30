import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models import get_model


DEFAULT_CONFIG = Path(__file__).parent / "config/model/dualtrack_ft_tus_rec_2025.yaml"
DEFAULT_CHECKPOINT = (
	Path(__file__).parent
	/ "experiments/dualtrack_ft_tus_rec_2025_v3_best.pt"
)


def load_frames(input_path):
	"""Load frames from either a processed DualTrack H5 or a raw TUS-REC H5."""
	with h5py.File(input_path, "r") as h5_file:
		key = "images" if "images" in h5_file else "frames"
		if key not in h5_file:
			raise KeyError(f"{input_path} must contain an 'images' or 'frames' dataset")
		frames = np.asarray(h5_file[key][:])

	if frames.ndim != 3:
		raise ValueError(f"Expected frames with shape [N, H, W], got {frames.shape}")
	return torch.from_numpy(frames).float().div(255.0)


def preprocess_frames(frames):
	"""Create the global [224x224] and local center-cropped [256x256] inputs."""
	frames = frames[:, None, :, :]
	global_images = F.interpolate(
		frames, size=(224, 224), mode="bilinear", align_corners=False
	)

	height, width = frames.shape[-2:]
	if height < 256 or width < 256:
		raise ValueError(
			f"Frames must be at least 256x256 for the local encoder, got {height}x{width}"
		)
	top = (height - 256) // 2
	left = (width - 256) // 2
	local_images = frames[:, :, top : top + 256, left : left + 256]
	return global_images, local_images


def load_model(config_path, checkpoint_path, device):
	cfg = OmegaConf.load(config_path)
	cfg.checkpoint = str(checkpoint_path)
	model = get_model(**cfg).to(device)
	model.eval()
	return model


def predict_streaming(model, global_images, local_images, window_size, device):
    """
    Streaming inference.

    Frames arrive one at a time.
    Once window_size frames are available, run inference.
    For every subsequent frame, remove the oldest frame and
    append the new frame.
    """

    global_buffer = []
    local_buffer = []
    predictions = []

    with torch.inference_mode():

        for frame_index, (global_frame, local_frame) in enumerate(zip(global_images, local_images)):

            # Add newly arrived frame
            global_buffer.append(global_frame)
            local_buffer.append(local_frame)

            print(f"Received frame {frame_index}")

            # Wait until 16 frames are available
            if len(global_buffer) < window_size:
                print(f"Waiting: {len(global_buffer)}/{window_size}")
                continue

            # Current 16-frame sliding window
            global_window = torch.stack(global_buffer, dim=0)
            local_window = torch.stack(local_buffer, dim=0)

            batch = {
                "sweep_id": [
                    f"live_{frame_index-window_size+1:04d}_{frame_index+1:04d}"
                ],
                "global_encoder_images": global_window.unsqueeze(0).to(device),
                "local_encoder_images": local_window.unsqueeze(0).to(device),
            }

            # Model input remains exactly the same
            window_prediction = model.predict(batch)[0]

            # Prediction corresponding to newest transition
            newest_prediction = window_prediction[-1]

            predictions.append(newest_prediction.detach().cpu())

            print(
                f"Window "
                f"[{frame_index-window_size+1}:{frame_index+1}]"
            )

            print(
                "Newest transition "
                f"{frame_index-1} -> {frame_index}:"
            )


            # Slide by ONE frame
            global_buffer.pop(0)
            local_buffer.pop(0)

        return torch.stack(predictions)


def save_predictions(predictions, output_path, window_size):
	with open(output_path, "w", newline="") as output_file:
		writer = csv.writer(output_file)
		writer.writerow(["frame_from", "frame_to", "tx", "ty", "tz", "rx", "ry", "rz"])
		for i, prediction in enumerate(predictions):
			frame_to = window_size - 1 + i
			frame_from = frame_to - window_size + 1
			writer.writerow([frame_from, frame_to, *prediction.tolist()])


def main():
	parser = argparse.ArgumentParser(description="Run DualTrack on a scan with sliding windows.")
	parser.add_argument("input_h5", type=Path, help="H5 file containing images or frames")
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
	parser.add_argument("--output-csv", type=Path, help="Save overlap-averaged predictions")
	parser.add_argument("--window-size", type=int, default=16)
	parser.add_argument("--stride", type=int, default=1)
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	args = parser.parse_args()

	if args.window_size < 2 or args.stride < 1:
		parser.error("--window-size must be >= 2 and --stride must be >= 1")

	device = torch.device(args.device)
	frames = load_frames(args.input_h5)
	global_images, local_images = preprocess_frames(frames)
	model = load_model(args.config, args.checkpoint, device)
	predictions = predict_streaming(
		model, global_images, local_images, args.window_size, device
	)

	if args.output_csv:
		save_predictions(predictions.detach().cpu(), args.output_csv, args.window_size)
		print(f"Saved {len(predictions)} transition predictions to {args.output_csv}")


if __name__ == "__main__":
	main()