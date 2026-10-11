import numpy as np

def fbm_rbf_reconstruct(points, intensities, voxel_size, max_control_points=20000,
                       neighbors=32, max_grid_voxels=5_000_000, chunk_size=50000):
    """Approximate function-based reconstruction using a local RBF interpolant.

    A continuous intensity function is fitted to a deterministic subset of samples,
    then evaluated on a regular 3D grid. Local neighbors and chunked evaluation
    limit memory/runtime compared with a global RBF fit.
    """
    from scipy.interpolate import RBFInterpolator

    points = np.asarray(points, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
    if len(points) == 0:
        return np.empty((0, 3)), np.empty((0,))
    if len(points) != len(intensities):
        raise ValueError("FBM: points and intensities must have equal lengths")

    # Deterministic, evenly spaced sample selection prevents quadratic global fitting.
    if len(points) > max_control_points:
        ids = np.linspace(0, len(points) - 1, max_control_points, dtype=np.int64)
        fit_points, fit_values = points[ids], intensities[ids]
    else:
        fit_points, fit_values = points, intensities

    # RBFInterpolator needs enough distinct samples for the chosen local neighborhood.
    local_neighbors = min(int(neighbors), len(fit_points))
    if local_neighbors < 4:
        raise ValueError("FBM requires at least 4 control points")
    interpolator = RBFInterpolator(
        fit_points, fit_values, kernel="thin_plate_spline", smoothing=1e-3,
        neighbors=local_neighbors, degree=1,
    )

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    axes = [np.arange(lo[d], hi[d] + voxel_size * 0.5, voxel_size) for d in range(3)]
    shape = tuple(len(axis) for axis in axes)
    total = int(np.prod(shape, dtype=np.int64))
    if total > max_grid_voxels:
        raise ValueError(
            f"FBM output grid has {total:,} voxels (limit {max_grid_voxels:,}). "
            "Increase --voxel-size or reduce the scan extent before running FBM."
        )

    # Use voxel-center coordinates and evaluate in chunks to avoid a large query array.
    axes = [lo[d] + (np.arange(shape[d], dtype=np.float64) + 0.5) * voxel_size
            for d in range(3)]
    flat_values = np.empty(total, dtype=np.float32)
    yz = shape[1] * shape[2]
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        flat_ids = np.arange(start, end, dtype=np.int64)
        ix = flat_ids // yz
        rem = flat_ids % yz
        iy = rem // shape[2]
        iz = rem % shape[2]
        query = np.column_stack((axes[0][ix], axes[1][iy], axes[2][iz]))
        flat_values[start:end] = interpolator(query).astype(np.float32)

    ii, jj, kk = np.indices(shape, dtype=np.int64)
    grid_points = np.column_stack((axes[0][ii.ravel()], axes[1][jj.ravel()],
                                   axes[2][kk.ravel()]))
    return grid_points, flat_values