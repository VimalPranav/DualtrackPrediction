import numpy as np

def voxel_nearest_neighbor(points, intensities, voxel_size, max_voxels=5_000_000):
    """Voxel-Based Method (VNN): assign every target voxel its nearest sample.

    Unlike Gaussian splatting, VNN does not smooth samples. It creates a regular
    grid over the sample bounding box and copies the intensity of the nearest
    input sample to every voxel center. This can fill unsupported regions, so
    use final-volume cropping and interpret filled areas cautiously.
    """
    points = np.asarray(points, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
    if len(points) == 0 or len(intensities) != len(points):
        return np.empty((0, 3)), np.empty((0,))
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    from scipy.spatial import cKDTree

    sample_voxels = np.floor(points / voxel_size).astype(np.int64)
    min_idx = sample_voxels.min(axis=0)
    max_idx = sample_voxels.max(axis=0)
    shape = (max_idx - min_idx + 1).astype(np.int64)
    total = int(np.prod(shape, dtype=np.int64))
    if total > max_voxels:
        raise ValueError(
            f"VNN grid would contain {total:,} voxels (limit {max_voxels:,}). "
            "Increase --voxel-size or use --method gaussian."
        )

    axes = [
        (np.arange(int(shape[d]), dtype=np.float64) + min_idx[d] + 0.5) * voxel_size
        for d in range(3)
    ]
    tree = cKDTree(points)
    flat_values = np.empty(total, dtype=np.float32)
    # Query in chunks to avoid allocating a second huge array of all grid points.
    chunk_size = 100_000
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        flat = np.arange(start, end, dtype=np.int64)
        ix, iy, iz = np.unravel_index(flat, tuple(shape.tolist()))
        query_points = np.column_stack((axes[0][ix], axes[1][iy], axes[2][iz]))
        _, nearest = tree.query(query_points, k=1, workers=-1)
        flat_values[start:end] = intensities[nearest]

    values = flat_values.reshape(tuple(shape.tolist()))
    grid_indices = np.indices(tuple(shape.tolist()), dtype=np.int32).reshape(3, -1).T
    global_indices = grid_indices + min_idx
    voxel_centers = (global_indices.astype(np.float64) + 0.5) * voxel_size
    return voxel_centers, values.reshape(-1).astype(np.float64)