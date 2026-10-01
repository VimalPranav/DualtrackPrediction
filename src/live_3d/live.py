import argparse
import csv
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from src.models import get_model
from src.live_3d.bin_filling import bin_fill


DEFAULT_CONFIG = Path(__file__).parent / "config/model/dualtrack_ft_tus_rec_2025.yaml"
DEFAULT_CHECKPOINT = (
	Path(__file__).parent
	/ "experiments/dualtrack_ft_tus_rec_2025_v3_best.pt"
)
DEFAULT_CALIBRATION = Path(__file__).parent / "calib_matrix.csv"


def load_frames(input_path):
	"""Load frames from either a processed DualTrack H5 or a raw TUS-REC H5."""
	with h5py.File(input_path, "r") as h5_file:
		key = "images" if "images" in h5_file else "frames"
		if key not in h5_file:
			raise KeyError(f"{input_path} must contain an 'images' or 'frames' dataset")
		
		frames = h5_file[key]
		if len(frames.shape) != 3:
			raise ValueError(f"Expected [N,H,W], got {frames.shape}")
		
		for frame_index in range(frames.shape[0]):
			yield frame_index, np.asarray(frames[frame_index])


def preprocess_frames(frame):
	"""Create the global [224x224] and local center-cropped [256x256] inputs."""
	frame = torch.from_numpy(frame).float()
	# Match the original code, but avoid dividing twice if an H5 is already float [0,1].
	if frame.max() > 1.0:
		frame = frame / 255.0
		
	if frame.ndim != 2:
		raise ValueError(f"Expected [H,W], got {tuple(frame.shape)}")
	
	frame = frame.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
	
	global_frame = F.interpolate(
        frame,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0]
	
	_, _, height, width = frame.shape
	
	if height < 256 or width < 256:
		raise ValueError(
            f"Frame must be at least 256x256, got {height}x{width}"
        )
	
	top = (height - 256) // 2
	left = (width - 256) // 2
	local_frame = frame[:, :, top:top + 256, left:left + 256][0]
	
	return global_frame, local_frame


def load_model(config_path, checkpoint_path, device):
	cfg = OmegaConf.load(config_path)
	cfg.checkpoint = str(checkpoint_path)
	model = get_model(**cfg).to(device)
	model.eval()
	return model

def load_calibration(calibration_path):
    """
    Read the attached calib_matrix.csv format:

      scaling_from_pixel_to_mm
      <4 rows>

      spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system
      <4 rows>

    Returns:
        pixel_to_mm: 4x4
        image_to_tool: 4x4
    """
    rows = []
    with open(calibration_path, "r", newline="") as f:
        for row in csv.reader(f):
            if row and any(cell.strip() for cell in row):
                rows.append([cell.strip() for cell in row])

    scaling_label = "scaling_from_pixel_to_mm"
    spatial_label = (
        "spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system"
    )

    scaling_index = next(
        (i for i, row in enumerate(rows) if row[0] == scaling_label), None
    )
    spatial_index = next(
        (i for i, row in enumerate(rows) if row[0] == spatial_label), None
    )

    if scaling_index is None or spatial_index is None:
        raise ValueError(
            "Calibration CSV does not contain the expected scaling/spatial labels."
        )

    pixel_to_mm = np.asarray(
        [[float(x) for x in rows[i]] for i in range(scaling_index + 1, scaling_index + 5)],
        dtype=np.float64,
    )
    image_to_tool = np.asarray(
        [[float(x) for x in rows[i]] for i in range(spatial_index + 1, spatial_index + 5)],
        dtype=np.float64,
    )

    if pixel_to_mm.shape != (4, 4) or image_to_tool.shape != (4, 4):
        raise ValueError("Expected two 4x4 calibration matrices.")

    return pixel_to_mm, image_to_tool

def pose_to_matrix(pose, rotation_order="xyz", degrees=False):
    """
    Convert [tx,ty,tz,rx,ry,rz] into a 4x4 rigid transform.

    IMPORTANT:
    The rotation convention MUST match the convention used by
    your DualTrack training targets.

    Translation units are preserved exactly as supplied by the model.
    """
    from scipy.spatial.transform import Rotation

    tx, ty, tz, rx, ry, rz = [float(x) for x in pose]

    R = Rotation.from_euler(
        rotation_order,
        [rx, ry, rz],
        degrees=degrees,
    ).as_matrix()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]

    return T

def image_pixels_to_tool(image, pixel_to_mm, image_to_tool, pixel_stride=4):
    """
    Sample pixels from one B-scan and map them through the supplied
    calibration matrices.

    Pixel homogeneous coordinate:
        [u, v, 0, 1]^T

    physical image coordinate:
        pixel_to_mm @ pixel

    calibrated tracking-tool coordinate:
        image_to_tool @ physical_image_coordinate
    """
    height, width = image.shape

    u = np.arange(0, width, pixel_stride, dtype=np.int64)
    v = np.arange(0, height, pixel_stride, dtype=np.int64)
    uu, vv = np.meshgrid(u, v)

    n = uu.size
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    pixel_points = np.stack(
        [
            uu.ravel().astype(np.float64) - cx,
            vv.ravel().astype(np.float64) - cy,
            np.zeros(n, dtype=np.float64),
            np.ones(n, dtype=np.float64),
        ],
        axis=1,
    )

    # Row-vector implementation of:
    # p_tool = image_to_tool @ pixel_to_mm @ p_pixel
    image_mm_points = pixel_points @ pixel_to_mm.T
    tool_points_h = image_mm_points @ image_to_tool.T

    intensities = image[vv, uu].astype(np.float64).ravel()
    return tool_points_h[:, :3], intensities

def transform_points(points, T):
    ones = np.ones((points.shape[0], 1), dtype=np.float64)

    homogeneous = np.concatenate(
        [points, ones],
        axis=1,
    )

    transformed = homogeneous @ T.T

    return transformed[:, :3]


def reconstruct_window(
    raw_frames,
    frame_poses,
    pixel_to_mm,
    image_to_tool,
    pixel_stride,
    voxel_size,
):
    """
    Reconstruct the CURRENT sliding window only.
    This is what is shown in the video-like Open3D viewer.
    """
    all_points = []
    all_intensities = []

    for image, T_world_tool in zip(raw_frames, frame_poses):
        tool_points, intensities = image_pixels_to_tool(
            image,
            pixel_to_mm,
            image_to_tool,
            pixel_stride=pixel_stride,
        )

        world_points = transform_points(tool_points, T_world_tool)

        all_points.append(world_points)
        all_intensities.append(intensities)

    all_points = np.concatenate(all_points, axis=0)
    all_intensities = np.concatenate(all_intensities, axis=0)

    return bin_fill(all_points, all_intensities, voxel_size)



class LivePointCloud:
    def __init__(self):
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "Open3D is required for visualization. Install it with: pip install open3d"
            ) from exc

        self.o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="Live 3D Ultrasound - Sliding Window Bin Filling",
            width=1100,
            height=800,
        )

        self.pcd = o3d.geometry.PointCloud()
        self.geometry_added = False
        self.first_update = True
        self.view_control = None
        self.fixed_front = None
        self.fixed_up = None
        self.fixed_zoom = None

    def update(self, points, intensities):
        if len(points) == 0:
            return

        values = intensities.astype(np.float64)
        lo = float(values.min())
        hi = float(values.max())

        if hi > lo:
            values = (values - lo) / (hi - lo)
        else:
            values = np.zeros_like(values)

        colors = np.column_stack((values, values, values))

        self.pcd.points = self.o3d.utility.Vector3dVector(points)
        self.pcd.colors = self.o3d.utility.Vector3dVector(colors)

        if not self.geometry_added:
            self.vis.add_geometry(self.pcd)
            self.geometry_added = True
        else:
            self.vis.update_geometry(self.pcd)

        # Fit the camera once. Subsequent windows update in the same world frame,
        # so the view does not continually reset.
        if self.first_update:
            self.vis.reset_view_point(True)
            self.view_control = self.vis.get_view_control()
            self.first_update = False
        else:
            center = np.asarray(points).mean(axis=0)
            self.view_control.set_lookat(center)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()



def run_live_reconstruction(
    input_h5,
    model,
    window_size,
    stride,
    device,
    pixel_to_mm,
    image_to_tool,
    pixel_stride,
    voxel_size,
    rotation_order,
    rotation_degrees,
    visualize=True,
):
    """
    Live-style sliding-window algorithm.

    Example for window_size=16, stride=4:
        [0:16]
        [4:20]
        [8:24]
        ...

    First window:
        use all 15 model outputs to initialize poses T0...T15.

    Later overlapping windows:
        poses for overlapping frames are already known.
        Use only the predictions needed to extend the trajectory to
        the NEW frames at the end of the window.

    The viewer always REPLACES the old point cloud with the current
    window's reconstruction, so it behaves like a video.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if stride > window_size:
        raise ValueError(
            "For continuous pose propagation, stride must be <= window_size."
        )

    # Buffers retain the current sliding window.
    global_buffer = deque(maxlen=window_size)
    local_buffer = deque(maxlen=window_size)
    raw_buffer = deque(maxlen=window_size)
    index_buffer = deque(maxlen=window_size)

    # One continuous absolute trajectory:
    # frame index -> 4x4 pose in frame-0/world coordinates.
    absolute_poses = {}

    viewer = LivePointCloud() if visualize else None

    # Number of frames that must arrive before the next inference.
    frames_since_last_inference = 0
    first_window_done = False

    try:
        with torch.inference_mode():
            for frame_index, raw_frame in load_frames(input_h5):
                global_frame, local_frame = preprocess_frames(raw_frame)

                global_buffer.append(global_frame)
                local_buffer.append(local_frame)
                raw_buffer.append(raw_frame)
                index_buffer.append(frame_index)

                print(
                    f"Received frame {frame_index} "
                    f"| buffer {len(index_buffer)}/{window_size}"
                )

                # Wait until the very first complete window exists.
                if len(index_buffer) < window_size:
                    continue

                if not first_window_done:
                    should_run = True
                else:
                    frames_since_last_inference += 1
                    should_run = frames_since_last_inference >= stride

                if not should_run:
                    continue

                window_indices = list(index_buffer)
                start_frame = window_indices[0]
                end_frame = window_indices[-1]

                global_images = torch.stack(list(global_buffer), dim=0)
                local_images = torch.stack(list(local_buffer), dim=0)

                batch = {
                    "sweep_id": [
                        f"live_{start_frame:04d}_{end_frame + 1:04d}"
                    ],
                    "global_encoder_images":
                        global_images.unsqueeze(0).to(device),
                    "local_encoder_images":
                        local_images.unsqueeze(0).to(device),
                }

                print(
                    f"\nDualTrack window [{start_frame}:{end_frame + 1}] "
                    f"(stride={stride})"
                )

                # For N input frames, DualTrack is expected to return N-1
                # consecutive relative transforms.
                window_prediction = model.predict(batch)[0]
                predictions = window_prediction.detach().cpu().numpy()

                if len(predictions) != window_size - 1:
                    raise RuntimeError(
                        f"Expected {window_size - 1} relative poses, "
                        f"got {len(predictions)}"
                    )

                # --------------------------------------------------------
                # FIRST WINDOW
                # --------------------------------------------------------
                if not first_window_done:
                    absolute_poses[start_frame] = np.eye(
                        4, dtype=np.float64
                    )

                    for j, prediction in enumerate(predictions):
                        frame_from = window_indices[j]
                        frame_to = window_indices[j + 1]

                        delta_T = pose_to_matrix(
                            prediction,
                            rotation_order=rotation_order,
                            degrees=rotation_degrees,
                        )

                        absolute_poses[frame_to] = (
                            absolute_poses[frame_from] @ delta_T
                        )

                    first_window_done = True

                # LATER OVERLAPPING WINDOWS
                #
                # Do NOT re-integrate overlapping transitions.
                # Only extend the trajectory when frame_to has not been
                # assigned an absolute pose yet.
                else:
                    new_pose_count = 0

                    for j, prediction in enumerate(predictions):
                        frame_from = window_indices[j]
                        frame_to = window_indices[j + 1]

                        if frame_to in absolute_poses:
                            continue

                        if frame_from not in absolute_poses:
                            raise RuntimeError(
                                f"Cannot extend pose to frame {frame_to}: "
                                f"pose for frame {frame_from} is unavailable. "
                                f"Use stride <= window_size."
                            )

                        delta_T = pose_to_matrix(
                            prediction,
                            rotation_order=rotation_order,
                            degrees=rotation_degrees,
                        )

                        absolute_poses[frame_to] = (
                            absolute_poses[frame_from] @ delta_T
                        )
                        new_pose_count += 1

                    print(
                        f"Added {new_pose_count} new frame poses "
                        f"to the global trajectory."
                    )

                # Every frame in the current window must now have a pose.
                missing = [
                    idx for idx in window_indices
                    if idx not in absolute_poses
                ]
                if missing:
                    raise RuntimeError(
                        f"Missing absolute poses for frames: {missing}"
                    )

                frame_poses = [
                    absolute_poses[idx] for idx in window_indices
                ]

                # BIN FILL CURRENT WINDOW
                cloud_points, cloud_intensities = reconstruct_window(
                    raw_frames=list(raw_buffer),
                    frame_poses=frame_poses,
                    pixel_to_mm=pixel_to_mm,
                    image_to_tool=image_to_tool,
                    pixel_stride=pixel_stride,
                    voxel_size=voxel_size,
                )

                print(
                    f"Bin-filled [{start_frame}:{end_frame + 1}] -> "
                    f"{len(cloud_points)} occupied voxels"
                )

                # Replace the displayed cloud: video-like sliding window.
                if viewer is not None:
                    viewer.update(cloud_points, cloud_intensities)

                frames_since_last_inference = 0

    finally:
        if viewer is not None:
            viewer.close()


def save_predictions(predictions, output_path):
	with open(output_path, "w", newline="") as output_file:
		writer = csv.writer(output_file)
		writer.writerow(["frame_from", "frame_to", "tx", "ty", "tz", "rx", "ry", "rz"])
		for frame_from, prediction in enumerate(predictions):
			writer.writerow([frame_from, frame_from + 1, *prediction.tolist()])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sliding-window DualTrack inference with calibrated "
            "bin-filled live 3D ultrasound reconstruction."
        )
    )

    parser.add_argument(
        "input_h5",
        type=Path,
        help="H5 file containing images or frames",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Calibration CSV containing the two 4x4 matrices",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Number of newly arrived frames between reconstructions",
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=4,
        help="Use every Nth image pixel for reconstruction",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="Voxel size in the same physical units as the reconstructed coordinates",
    )
    parser.add_argument(
        "--rotation-order",
        default="xyz",
        help="Euler order for rx,ry,rz; must match DualTrack targets",
    )
    parser.add_argument(
        "--rotation-degrees",
        action="store_true",
        help="Interpret rx,ry,rz as degrees instead of radians",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )

    args = parser.parse_args()

    if args.window_size < 2:
        parser.error("--window-size must be >= 2")
    if args.stride < 1 or args.stride > args.window_size:
        parser.error("--stride must satisfy 1 <= stride <= window-size")
    if args.pixel_stride < 1:
        parser.error("--pixel-stride must be >= 1")
    if args.voxel_size <= 0:
        parser.error("--voxel-size must be > 0")

    device = torch.device(args.device)

    pixel_to_mm, image_to_tool = load_calibration(args.calibration)

    print(f"Device: {device}")
    print(f"Window size: {args.window_size}")
    print(f"Sliding stride: {args.stride}")
    print(f"Pixel stride: {args.pixel_stride}")
    print(f"Voxel size: {args.voxel_size}")
    print("\nPixel -> mm calibration:")
    print(pixel_to_mm)
    print("\nImage -> tracking-tool calibration:")
    print(image_to_tool)

    model = load_model(
        args.config,
        args.checkpoint,
        device,
    )

    run_live_reconstruction(
        input_h5=args.input_h5,
        model=model,
        window_size=args.window_size,
        stride=args.stride,
        device=device,
        pixel_to_mm=pixel_to_mm,
        image_to_tool=image_to_tool,
        pixel_stride=args.pixel_stride,
        voxel_size=args.voxel_size,
        rotation_order=args.rotation_order,
        rotation_degrees=args.rotation_degrees,
        visualize=not args.no_viewer,
    )

if __name__ == "__main__":
	main()