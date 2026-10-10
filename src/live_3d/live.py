import argparse
import csv
import os
import time
import threading
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models import get_model
from src.live_3d.bin_filling import bin_fill
from src.live_3d.gaussian_splatting import gaussian_weighted_splat
from src.live_3d.VNN import voxel_nearest_neighbor
from src.live_3d.fbm import fbm_rbf_reconstruct


DEFAULT_CONFIG = Path(__file__).parent / "config/model/dualtrack_ft_tus_rec_2025.yaml"
DEFAULT_CHECKPOINT = (
	Path(__file__).parent
	/ "experiments/dualtrack_ft_tus_rec_2025_v3_best.pt"
)
DEFAULT_CALIBRATION = Path(__file__).parent / "calib_matrix.csv"




def measure_grid_coverage(sample_points, voxel_size, support_radius_mm, max_grid_voxels=5_000_000):
    """Measure sample-supported coverage on a fixed grid, independent of method output."""
    from scipy.spatial import cKDTree

    points = np.asarray(sample_points, dtype=np.float64)
    if points.size == 0:
        return {"grid_voxels": 0, "supported_voxels": 0,
                "coverage_percent": 0.0, "hole_ratio_percent": 100.0}
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    shape = np.maximum(1, np.ceil((hi - lo) / voxel_size).astype(np.int64))
    total = int(np.prod(shape, dtype=np.int64))
    if total > max_grid_voxels:
        raise ValueError(
            f"Coverage grid would contain {total:,} voxels (limit {max_grid_voxels:,}). "
            "Use a larger --voxel-size or increase the limit in the script."
        )
    axes = [lo[d] + (np.arange(shape[d]) + 0.5) * voxel_size for d in range(3)]
    tree = cKDTree(points)
    supported_count = 0
    # Chunk along X to avoid allocating all grid coordinates at once.
    for ix, x in enumerate(axes[0]):
        yy, zz = np.meshgrid(axes[1], axes[2], indexing="ij")
        grid = np.column_stack((np.full(yy.size, x), yy.ravel(), zz.ravel()))
        distances, _ = tree.query(grid, k=1, workers=-1)
        supported_count += int(np.count_nonzero(distances <= support_radius_mm))
    coverage = 100.0 * supported_count / total
    return {"grid_voxels": total, "supported_voxels": supported_count,
            "coverage_percent": coverage, "hole_ratio_percent": 100.0 - coverage}


def save_benchmark_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def crop_final_volume_to_bright_region(
    points, intensities, threshold_quantile=0.85, margin_fraction=0.08
):
    """Crop the FINAL volume around brighter ultrasound structures.

    Bright voxels determine the crop bounds, but all voxels inside those bounds
    are retained so darker internal anatomy is not automatically discarded.
    This function is intended for the final volume only, not live windows.
    """
    points = np.asarray(points)
    intensities = np.asarray(intensities)

    if len(points) == 0 or len(intensities) != len(points):
        return points, intensities

    threshold = np.quantile(intensities, threshold_quantile)
    seed_mask = intensities >= threshold
    seed_points = points[seed_mask]
    if len(seed_points) == 0:
        return points, intensities

    # Bounding box around stronger echoes. Add a margin to preserve nearby tissue.
    lo = seed_points.min(axis=0)
    hi = seed_points.max(axis=0)
    full_extent = points.max(axis=0) - points.min(axis=0)
    margin = np.maximum(full_extent * margin_fraction, 1e-6)
    lo = lo - margin
    hi = hi + margin

    keep = np.all((points >= lo) & (points <= hi), axis=1)
    if not np.any(keep):
        return points, intensities
    return points[keep], intensities[keep]

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

def reconstruct_samples(points, intensities, method, voxel_size,
                        kernel_sigma_mm, kernel_truncate, min_kernel_weight):
    """Dispatch to Gaussian splatting or VNN for a fair same-input comparison."""
    if method == "vnn":
        return voxel_nearest_neighbor(points, intensities, voxel_size)
    if method == "gaussian":
        return gaussian_weighted_splat(
            points=points,
            intensities=intensities,
            voxel_size=voxel_size,
            sigma_mm=kernel_sigma_mm,
            truncate=kernel_truncate,
            min_weight=min_kernel_weight,
        )
    if method == "fbm":
        return fbm_rbf_reconstruct(points, intensities, voxel_size)
        
    raise ValueError(f"Unknown reconstruction method: {method}")


def reconstruct_window(
    raw_frames,
    frame_poses,
    pixel_to_mm,
    image_to_tool,
    pixel_stride,
    voxel_size,
    kernel_sigma_mm,
    kernel_truncate,
    min_kernel_weight,
    method="gaussian",
    timings=None,
):
    """
    Reconstruct the CURRENT sliding window only.
    This is what is shown in the video-like Open3D viewer.
    """
    all_points = []
    all_intensities = []

    for image, T_world_tool in zip(raw_frames, frame_poses):
        _point_t0 = time.perf_counter()
        tool_points, intensities = image_pixels_to_tool(
            image,
            pixel_to_mm,
            image_to_tool,
            pixel_stride=pixel_stride,
        )

        world_points = transform_points(tool_points, T_world_tool)
        if timings is not None:
            timings["point_generation_seconds"] += time.perf_counter() - _point_t0

        all_points.append(world_points)
        all_intensities.append(intensities)

    if not all_points:
        return np.empty((0, 3)), np.empty((0,))

    all_points = np.concatenate(all_points, axis=0)
    all_intensities = np.concatenate(all_intensities, axis=0)

    _recon_t0 = time.perf_counter()
    result = reconstruct_samples(
        points=all_points,
        intensities=all_intensities,
        method=method,
        voxel_size=voxel_size,
        kernel_sigma_mm=kernel_sigma_mm,
        kernel_truncate=kernel_truncate,
        min_kernel_weight=min_kernel_weight,
    )
    if timings is not None:
        timings["live_reconstruction_seconds"] += time.perf_counter() - _recon_t0
    return result


class PersistentGaussianVolume:
    """
    Sparse-to-dense whole-scan reconstruction accumulator.

    Each newly observed frame contributes to a persistent voxel grid exactly
    once. Gaussian filtering is applied to the persistent intensity and
    support grids when a visualization update is requested.

    This is different from the sliding-window viewer: the displayed volume
    grows with the scan instead of being replaced by the newest 16 frames.
    """
    def __init__(self, voxel_size):
        self.voxel_size = float(voxel_size)
        self.intensity_sum = None
        self.sample_weight = None
        self.min_idx = None
        self.frame_ids = set()
        self._sample_points = []
        self._sample_intensities = []

    def add_frame_samples(self, frame_id, points, intensities):
        if frame_id in self.frame_ids or len(points) == 0:
            return False

        # Preserve original samples as well as the Gaussian accumulator so the
        # final VNN reconstruction can use the same input samples.
        self._sample_points.append(np.asarray(points, dtype=np.float32).copy())
        self._sample_intensities.append(np.asarray(intensities, dtype=np.float32).copy())

        voxel_indices = np.floor(points / self.voxel_size).astype(np.int64)
        new_min = voxel_indices.min(axis=0)
        new_max = voxel_indices.max(axis=0)

        if self.intensity_sum is None:
            self.min_idx = new_min.copy()
            max_idx = new_max.copy()
            shape = tuple((max_idx - self.min_idx + 1).tolist())
            self.intensity_sum = np.zeros(shape, dtype=np.float32)
            self.sample_weight = np.zeros(shape, dtype=np.float32)
        else:
            old_min = self.min_idx.copy()
            old_shape = np.asarray(self.intensity_sum.shape, dtype=np.int64)
            old_max = old_min + old_shape - 1

            expanded_min = np.minimum(old_min, new_min)
            expanded_max = np.maximum(old_max, new_max)

            if np.any(expanded_min != old_min) or np.any(expanded_max != old_max):
                new_shape = tuple((expanded_max - expanded_min + 1).tolist())
                new_sum = np.zeros(new_shape, dtype=np.float32)
                new_weight = np.zeros(new_shape, dtype=np.float32)

                offset = old_min - expanded_min
                old_slices = tuple(
                    slice(int(offset[d]), int(offset[d] + old_shape[d]))
                    for d in range(3)
                )
                new_sum[old_slices] = self.intensity_sum
                new_weight[old_slices] = self.sample_weight

                self.intensity_sum = new_sum
                self.sample_weight = new_weight
                self.min_idx = expanded_min

        local = voxel_indices - self.min_idx
        shape = self.intensity_sum.shape
        flat = np.ravel_multi_index(local.T, dims=shape)
        n = int(np.prod(shape))

        self.intensity_sum += np.bincount(
            flat,
            weights=intensities.astype(np.float32),
            minlength=n,
        ).reshape(shape).astype(np.float32)
        self.sample_weight += np.bincount(
            flat,
            weights=np.ones(len(points), dtype=np.float32),
            minlength=n,
        ).reshape(shape).astype(np.float32)

        self.frame_ids.add(int(frame_id))
        return True

    def reconstruct(self, sigma_mm, truncate, min_weight, method="gaussian"):
        if method == "vnn":
            if not self._sample_points:
                return np.empty((0, 3)), np.empty((0,))
            all_points = np.concatenate(self._sample_points, axis=0)
            all_intensities = np.concatenate(self._sample_intensities, axis=0)
            return voxel_nearest_neighbor(all_points, all_intensities, self.voxel_size)
        if method == "fbm":
            if not self._sample_points:
                return np.empty((0, 3)), np.empty((0,))
            all_points = np.concatenate(self._sample_points, axis=0)
            all_intensities = np.concatenate(self._sample_intensities, axis=0)
            return fbm_rbf_reconstruct(all_points, all_intensities, self.voxel_size)
        if method != "gaussian":
            raise ValueError(f"Unknown reconstruction method: {method}")
        if self.intensity_sum is None:
            return np.empty((0, 3)), np.empty((0,))

        from scipy.ndimage import gaussian_filter

        sigma_vox = float(sigma_mm) / self.voxel_size
        blurred_intensity = gaussian_filter(
            self.intensity_sum,
            sigma=sigma_vox,
            mode="constant",
            cval=0.0,
            truncate=float(truncate),
        )
        blurred_weight = gaussian_filter(
            self.sample_weight,
            sigma=sigma_vox,
            mode="constant",
            cval=0.0,
            truncate=float(truncate),
        )

        valid = blurred_weight > float(min_weight)
        if not np.any(valid):
            return np.empty((0, 3)), np.empty((0,))

        values = np.zeros_like(blurred_intensity)
        values[valid] = blurred_intensity[valid] / blurred_weight[valid]

        idx = np.argwhere(valid)
        global_idx = idx + self.min_idx
        points = (global_idx.astype(np.float64) + 0.5) * self.voxel_size
        return points, values[valid].astype(np.float64)


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
            window_name="Live 3D Ultrasound - 360 View (Drag to Rotate)",
            width=1100,
            height=800,
        )

        # Open3D's native mouse controls stay enabled during live updates:
        # left-drag rotates, mouse wheel zooms, right-drag pans.
        self.pcd = o3d.geometry.PointCloud()
        self.geometry_added = False
        self.first_update = True
        self.view_control = None
        self.auto_rotate = False
        self.rotation_step_deg = 3.0
        self.orbit_angle_deg = 0.0
        self.current_center = np.zeros(3, dtype=np.float64)
        self.closed = False

    def set_auto_rotate(self, enabled=True, step=3.0):
        self.auto_rotate = bool(enabled)
        self.rotation_step_deg = float(step)

    def _set_cloud(self, points, intensities):
        """Replace the displayed geometry with a new reconstructed cloud."""
        if len(points) == 0:
            return False

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
        self.current_center = np.asarray(points).mean(axis=0)

        if not self.geometry_added:
            self.vis.add_geometry(self.pcd)
            self.geometry_added = True
        else:
            self.vis.update_geometry(self.pcd)

        return True

    def _update_camera(self, reset=False):
        """Maintain a usable camera without relying on unsupported getters."""
        if self.view_control is None:
            self.view_control = self.vis.get_view_control()

        if reset:
            # Open3D chooses the initial orientation and zoom.
            self.vis.reset_view_point(True)
            self.view_control = self.vis.get_view_control()
            self.view_control.set_lookat(self.current_center)
            self.first_update = False
            return

        # Follow the cloud center while retaining the current camera setup.
        self.view_control.set_lookat(self.current_center)

        if self.auto_rotate:
            self.orbit_angle_deg = (
                self.orbit_angle_deg + self.rotation_step_deg
            ) % 360.0
            theta = np.deg2rad(self.orbit_angle_deg)
            front = np.array([
                np.sin(theta),
                0.0,
                np.cos(theta),
            ], dtype=np.float64)
            self.view_control.set_front(front)
            self.view_control.set_up([0.0, 1.0, 0.0])

    def update(self, points, intensities, phase_label="LIVE"):
        """
        Replace the visible cloud and process GUI events once.

        phase_label is printed only for clarity in the terminal.
        """
        if self.closed:
            return False

        if len(points) == 0:
            self.vis.poll_events()
            self.vis.update_renderer()
            return True

        self._set_cloud(points, intensities)
        # Only initialize the camera on the first frame. Later updates replace
        # geometry but preserve the user's manual 360-degree viewing angle.
        self._update_camera(reset=self.first_update)

        alive = self.vis.poll_events()
        self.vis.update_renderer()
        if alive is False:
            self.closed = True
            return False

        print(f"Viewer: {phase_label} | {len(points)} displayed voxels")
        print("View controls: left-drag = rotate 360 | scroll = zoom | right-drag = pan")
        return True

    def show_final_until_closed(self, points, intensities):
        """
        Replace the live cloud with the COMPLETE accumulated reconstruction,
        label the terminal phase, and keep the Open3D window alive until the
        user closes it manually.
        """
        if self.closed:
            return

        if len(points) == 0:
            print("Final volume is empty; nothing to display.")
            return

        self._set_cloud(points, intensities)
        self._update_camera(reset=self.first_update)

        print("\n============================================================")
        print("FINAL STATIC 3D VOLUME")
        print(f"Displayed voxels: {len(points)}")
        print("The complete accumulated reconstruction is now shown.")
        print("View controls: left-drag = rotate 360 | scroll = zoom | right-drag = pan.")
        print("Close the Open3D window to exit.")
        print("============================================================")

        while not self.closed:
            alive = self.vis.poll_events()
            self.vis.update_renderer()

            if alive is False:
                self.closed = True
                break

            if self.auto_rotate:
                self.orbit_angle_deg = (
                    self.orbit_angle_deg + self.rotation_step_deg
                ) % 360.0
                theta = np.deg2rad(self.orbit_angle_deg)
                front = np.array([
                    np.sin(theta),
                    0.0,
                    np.cos(theta),
                ], dtype=np.float64)
                self.view_control.set_front(front)
                self.view_control.set_up([0.0, 1.0, 0.0])

    def close(self):
        if not self.closed:
            self.vis.destroy_window()
            self.closed = True



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
    kernel_sigma_mm,
    kernel_truncate,
    min_kernel_weight,
    method,
    rotation_order,
    rotation_degrees,
    accumulate=True,
    auto_rotate=False,
    rotation_step=3.0,
    visualize=True,
    benchmark_csv="benchmark_results.csv",
    support_radius_mm=1.5,
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

    global_volume = PersistentGaussianVolume(voxel_size) if accumulate else None

    viewer = LivePointCloud() if visualize else None
    if viewer is not None:
        viewer.set_auto_rotate(auto_rotate, rotation_step)

    # Number of frames that must arrive before the next inference.
    frames_since_last_inference = 0
    first_window_done = False
    last_processed_end_frame = None

    # Benchmark accumulators. Peak RSS is sampled in the background during processing.
    benchmark_start = time.perf_counter()
    timings = {"inference_seconds": 0.0, "live_reconstruction_seconds": 0.0,
               "final_reconstruction_seconds": 0.0, "point_generation_seconds": 0.0}
    memory_samples = {"peak_rss_mb": 0.0, "peak_cpu_percent": 0.0,
                      "cpu_percent_sum": 0.0, "cpu_samples": 0}
    sampler_stop = threading.Event()
    try:
        import psutil
        process = psutil.Process(os.getpid())
        process.cpu_percent(interval=None)  # Prime the CPU utilization counter.
        def sample_memory():
            while not sampler_stop.is_set():
                try:
                    rss = process.memory_info().rss / (1024 ** 2)
                    cpu_pct = process.cpu_percent(interval=None)
                    memory_samples["peak_rss_mb"] = max(memory_samples["peak_rss_mb"], rss)
                    memory_samples["peak_cpu_percent"] = max(memory_samples["peak_cpu_percent"], cpu_pct)
                    memory_samples["cpu_percent_sum"] += cpu_pct
                    memory_samples["cpu_samples"] += 1
                except Exception:
                    pass
                sampler_stop.wait(0.1)
        sampler_thread = threading.Thread(target=sample_memory, daemon=True)
        sampler_thread.start()
    except ImportError:
        process = None
        sampler_thread = None
        print("Benchmark note: install psutil for CPU/RAM measurements: python -m pip install psutil")

    process_elapsed_before_final_view = None
    final_metric_row = None

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
                    _t0 = time.perf_counter()
                    window_prediction = model.predict(batch)[0]
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    timings["inference_seconds"] += time.perf_counter() - _t0
                    predictions = window_prediction.detach().cpu().numpy()
    
                    if len(predictions) != window_size - 1:
                        raise RuntimeError(
                            f"Expected {window_size - 1} relative poses, "
                            f"got {len(predictions)}"
                        )
    
                    # FIRST WINDOW
               
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
    
                    # 3D RECONSTRUCTION
                    #   1. Reconstruct the CURRENT 16-frame window for the live viewer.
                    #   2. In parallel, add only NEW frames to the persistent global volume.
    
                    current_cloud_points, current_cloud_intensities = reconstruct_window(
                        raw_frames=list(raw_buffer),
                        frame_poses=frame_poses,
                        pixel_to_mm=pixel_to_mm,
                        image_to_tool=image_to_tool,
                        pixel_stride=pixel_stride,
                        voxel_size=voxel_size,
                        kernel_sigma_mm=kernel_sigma_mm,
                        kernel_truncate=kernel_truncate,
                        min_kernel_weight=min_kernel_weight,
                        method=method,
                        timings=timings,
                    )
    
                    print(
                        f"Live {method.upper()} reconstruction [{start_frame}:{end_frame + 1}] -> "
                        f"{len(current_cloud_points)} voxels"
                    )
    
                    # Always accumulate the complete scan. This is intentionally
                    # separate from the currently displayed sliding window.
                    new_frame_count = 0
                    if global_volume is not None:
                        for idx, image, pose in zip(
                            window_indices, list(raw_buffer), frame_poses
                        ):
                            if idx in global_volume.frame_ids:
                                continue
    
                            _point_t0 = time.perf_counter()
                            tool_points, intensities = image_pixels_to_tool(
                                image,
                                pixel_to_mm,
                                image_to_tool,
                                pixel_stride=pixel_stride,
                            )
                            world_points = transform_points(tool_points, pose)
                            timings["point_generation_seconds"] += time.perf_counter() - _point_t0
                            global_volume.add_frame_samples(
                                idx, world_points, intensities
                            )
                            new_frame_count += 1
    
                        print(
                            f"Accumulated global volume: "
                            f"{len(global_volume.frame_ids)} frames total "
                            f"({new_frame_count} new frames added)"
                        )
    
                    # During acquisition ONLY the current sliding-window cloud is shown.
                    if viewer is not None:
                        alive = viewer.update(
                            current_cloud_points,
                            current_cloud_intensities,
                            phase_label=(
                                f"LIVE window [{start_frame}:{end_frame + 1}]"
                            ),
                        )
                        if not alive:
                            print("Viewer closed by user. Stopping live reconstruction.")
                            return
    
                    last_processed_end_frame = end_frame
                    frames_since_last_inference = 0
    
            # END-OF-STREAM FLUSH

            # If the total frame count is not aligned with the chosen stride, the
            # final few frames may not have triggered a normal sliding-window
            # inference. Run the last complete 16-frame window once so that frames
            # up to the actual final H5 frame also receive poses and enter the
            # persistent reconstruction.
            if first_window_done and len(index_buffer) == window_size:
                final_window_indices = list(index_buffer)
                final_end_frame = final_window_indices[-1]
    
                if last_processed_end_frame != final_end_frame:
                    final_start_frame = final_window_indices[0]
                    print(
                        f"\nFinal DualTrack flush window "
                        f"[{final_start_frame}:{final_end_frame + 1}]"
                    )
    
                    final_global_images = torch.stack(
                        list(global_buffer), dim=0
                    )
                    final_local_images = torch.stack(
                        list(local_buffer), dim=0
                    )
    
                    final_batch = {
                        "sweep_id": [
                            f"final_{final_start_frame:04d}_{final_end_frame + 1:04d}"
                        ],
                        "global_encoder_images":
                            final_global_images.unsqueeze(0).to(device),
                        "local_encoder_images":
                            final_local_images.unsqueeze(0).to(device),
                    }
    
                    _infer_t0 = time.perf_counter()
                    with torch.inference_mode():
                        final_prediction = model.predict(final_batch)[0]
                    if str(device).startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    timings["inference_seconds"] += time.perf_counter() - _infer_t0
                    final_predictions = (
                        final_prediction.detach().cpu().numpy()
                    )
    
                    if len(final_predictions) != window_size - 1:
                        raise RuntimeError(
                            f"Final flush expected {window_size - 1} relative poses, "
                            f"got {len(final_predictions)}"
                        )
    
                    # Extend only poses that do not already exist.
                    for j, prediction in enumerate(final_predictions):
                        frame_from = final_window_indices[j]
                        frame_to = final_window_indices[j + 1]
    
                        if frame_to in absolute_poses:
                            continue
                        if frame_from not in absolute_poses:
                            raise RuntimeError(
                                f"Cannot extend final pose to frame {frame_to}: "
                                f"pose for frame {frame_from} is unavailable."
                            )
    
                        delta_T = pose_to_matrix(
                            prediction,
                            rotation_order=rotation_order,
                            degrees=rotation_degrees,
                        )
                        absolute_poses[frame_to] = (
                            absolute_poses[frame_from] @ delta_T
                        )
    
                    # Add any frames from the final window that were not already
                    # inserted into the persistent volume.
                    if global_volume is not None:
                        final_frame_poses = [
                            absolute_poses[idx]
                            for idx in final_window_indices
                        ]
                        flushed_count = 0
    
                        for idx, image, pose in zip(
                            final_window_indices,
                            list(raw_buffer),
                            final_frame_poses,
                        ):
                            if idx in global_volume.frame_ids:
                                continue
    
                            tool_points, intensities = image_pixels_to_tool(
                                image,
                                pixel_to_mm,
                                image_to_tool,
                                pixel_stride=pixel_stride,
                            )
                            world_points = transform_points(tool_points, pose)
                            global_volume.add_frame_samples(
                                idx, world_points, intensities
                            )
                            flushed_count += 1
    
                        print(
                            f"Final flush added {flushed_count} frame(s); "
                            f"global volume now contains "
                            f"{len(global_volume.frame_ids)} frames"
                        )
    
            # END OF STREAM: SWITCH FROM LIVE WINDOW TO FINAL VOLUME
            if viewer is not None and global_volume is not None and not viewer.closed:
                print("\nInput stream finished.")
                print("Reconstructing the complete accumulated volume...")
    
                _final_recon_t0 = time.perf_counter()
                final_points, final_intensities = global_volume.reconstruct(
                    sigma_mm=kernel_sigma_mm,
                    truncate=kernel_truncate,
                    min_weight=min_kernel_weight,
                    method=method,
                )
                timings["final_reconstruction_seconds"] += time.perf_counter() - _final_recon_t0
    
                print(
                    f"Final global {method.upper()} reconstruction: "
                    f"{len(global_volume.frame_ids)} frames -> "
                    f"{len(final_points)} voxels"
                )
    
                # Crop ONLY the final volume around brighter ultrasound structures.
                # Live sliding-window reconstructions are left unchanged.
                before_crop = len(final_points)
                final_points, final_intensities = crop_final_volume_to_bright_region(
                    final_points,
                    final_intensities,
                    threshold_quantile=0.85,  # top 15% intensities define the ROI
                    margin_fraction=0.08,     # preserve 8% of full extent around it
                )
                print(
                    f"Final-volume crop: {before_crop} -> {len(final_points)} voxels"
                )

                # Metrics are computed from the original sample points, not the cropped display cloud.
                if global_volume is not None and global_volume._sample_points:
                    all_sample_points = np.concatenate(global_volume._sample_points, axis=0)
                    try:
                        coverage_metrics = measure_grid_coverage(
                            all_sample_points, voxel_size, support_radius_mm
                        )
                    except ValueError as exc:
                        print(f"Coverage metric skipped: {exc}")
                        coverage_metrics = {"grid_voxels": 0, "supported_voxels": 0,
                                            "coverage_percent": float("nan"),
                                            "hole_ratio_percent": float("nan")}
                else:
                    coverage_metrics = {"grid_voxels": 0, "supported_voxels": 0,
                                        "coverage_percent": float("nan"), "hole_ratio_percent": float("nan")}

                process_elapsed_before_final_view = time.perf_counter() - benchmark_start
                total_frames = len(global_volume.frame_ids) if global_volume is not None else len(absolute_poses)
                total_runtime = process_elapsed_before_final_view
                peak_ram = memory_samples["peak_rss_mb"]
                gpu_peak_mb = float("nan")
                if str(device).startswith("cuda") and torch.cuda.is_available():
                    gpu_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                final_metric_row = {
                    "input_file": str(input_h5), "method": method, "frames": total_frames,
                    "voxel_size": voxel_size, "pixel_stride": pixel_stride,
                    "support_radius_mm": support_radius_mm,
                    "inference_seconds": timings["inference_seconds"],
                    "point_generation_seconds": timings["point_generation_seconds"],
                    "live_reconstruction_seconds_total": timings["live_reconstruction_seconds"],
                    "final_reconstruction_seconds": timings["final_reconstruction_seconds"],
                    "processing_runtime_seconds_excluding_final_view": total_runtime,
                    "peak_rss_mb_sampled": peak_ram,
                    "peak_cpu_percent_sampled": memory_samples["peak_cpu_percent"],
                    "mean_cpu_percent_sampled": (memory_samples["cpu_percent_sum"] / memory_samples["cpu_samples"] if memory_samples["cpu_samples"] else float("nan")),
                    "peak_gpu_allocated_mb": gpu_peak_mb,
                    "grid_voxels": coverage_metrics["grid_voxels"],
                    "supported_voxels": coverage_metrics["supported_voxels"],
                    "coverage_percent": coverage_metrics["coverage_percent"],
                    "hole_ratio_percent": coverage_metrics["hole_ratio_percent"],
                    "final_voxels_before_crop": before_crop,
                    "final_voxels_after_crop": len(final_points),
                }
                save_benchmark_csv(benchmark_csv, final_metric_row)
                print("\n========== BENCHMARK SUMMARY ==========")
                for key, value in final_metric_row.items():
                    print(f"{key}: {value}")
                print(f"Saved benchmark row to: {benchmark_csv}")
                print("=======================================\n")

                # Keep the final cropped volume open until manually closed.
                viewer.show_final_until_closed(
                    final_points,
                    final_intensities,
                )

    finally:
        sampler_stop.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=1.0)
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
            "Sliding-window DualTrack inference with interactive 360-degree "
            "live 3D ultrasound reconstruction."
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
        "--method",
        choices=("gaussian", "vnn", "fbm"),
        default="gaussian",
        help=("Reconstruction method: gaussian = normalized Gaussian splatting; "
              "vnn = Voxel-Based Nearest Neighbor, assigning every grid voxel "
              "the nearest sample intensity."),
    )
    parser.add_argument(
        "--kernel-sigma-mm",
        type=float,
        default=1.0,
        help=(
            "Gaussian reconstruction bandwidth in mm. Larger values "
            "fill larger inter-slice gaps but produce more smoothing."
        ),
    )
    parser.add_argument(
        "--kernel-truncate",
        type=float,
        default=3.0,
        help="Gaussian kernel radius in multiples of sigma.",
    )
    parser.add_argument(
        "--min-kernel-weight",
        type=float,
        default=0.02,
        help=(
            "Minimum normalized Gaussian support required to keep a voxel. "
            "Lower values fill farther into gaps but may create unsupported haze."
        ),
    )
    parser.add_argument(
        "--no-final-volume",
        action="store_true",
        help=(
            "Disable the final whole-scan volume and close after the last "
            "live sliding-window reconstruction. By default the complete "
            "0..last-frame volume is shown after acquisition."
        ),
    )
    parser.add_argument(
        "--auto-rotate",
        action="store_true",
        help="Continuously rotate the Open3D view while reconstruction runs.",
    )
    parser.add_argument(
        "--rotation-step",
        type=float,
        default=3.0,
        help="Automatic orbit increment in degrees per reconstruction update.",
    )
    parser.add_argument(
        "--rotation-order",
        default="xyz",
        help="Euler order for rx,ry,rz; must match DualTrack targets",
    )
    parser.add_argument(
        "--rotation-radians",
        action="store_true",
        help="Interpret rx,ry,rz as radians. By default DualTrack angles are treated as degrees.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )

    parser.add_argument(
        "--benchmark-csv", type=Path, default=Path("benchmark_results.csv"),
        help="CSV file to append runtime, memory, coverage, and hole metrics to",
    )
    parser.add_argument(
        "--support-radius-mm", type=float, default=1.5,
        help="Fixed distance threshold for sample-supported coverage (mm)",
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
    if args.support_radius_mm <= 0:
        parser.error("--support-radius-mm must be > 0")
    if args.kernel_sigma_mm <= 0:
        parser.error("--kernel-sigma-mm must be > 0")
    if args.kernel_truncate <= 0:
        parser.error("--kernel-truncate must be > 0")
    if not 0.0 <= args.min_kernel_weight:
        parser.error("--min-kernel-weight must be >= 0")
    if args.rotation_step < 0:
        parser.error("--rotation-step must be >= 0")

    device = torch.device(args.device)

    pixel_to_mm, image_to_tool = load_calibration(args.calibration)

    print(f"Device: {device}")
    print(f"Window size: {args.window_size}")
    print(f"Sliding stride: {args.stride}")
    print(f"Pixel stride: {args.pixel_stride}")
    print(f"Voxel size: {args.voxel_size}")
    print(f"Reconstruction method: {args.method}")
    print(f"Gaussian sigma: {args.kernel_sigma_mm} mm")
    print(f"Gaussian truncate: {args.kernel_truncate} sigma")
    print(f"Minimum kernel weight: {args.min_kernel_weight}")
    print(f"Final whole-scan volume: {not args.no_final_volume}")
    print(f"Auto rotation: {args.auto_rotate}")
    print("\nPixel -> mm calibration:")
    print(pixel_to_mm)
    print("\nImage -> tracking-tool calibration:")
    print(image_to_tool)

    model = load_model(
        args.config,
        args.checkpoint,
        device,
    )
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

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
        kernel_sigma_mm=args.kernel_sigma_mm,
        kernel_truncate=args.kernel_truncate,
        min_kernel_weight=args.min_kernel_weight,
        method=args.method,
        rotation_order=args.rotation_order,
        rotation_degrees=not args.rotation_radians,
        accumulate=not args.no_final_volume,
        auto_rotate=args.auto_rotate,
        rotation_step=args.rotation_step,
        visualize=not args.no_viewer,
        benchmark_csv=args.benchmark_csv,
        support_radius_mm=args.support_radius_mm,
    )
    
if __name__ == "__main__":
	main()