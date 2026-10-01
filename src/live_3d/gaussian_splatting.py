import numpy as np

def gaussian_weighted_splat(
    points,
    intensities,
    voxel_size,
    sigma_mm=1.0,
    truncate=3.0,
    min_weight=0.02,
):
    """
    Reconstruct a dense 3D ultrasound volume using normalized
    Gaussian-weighted splatting.

    The input is a scattered set of samples:
        points[i]      = [X, Y, Z]
        intensities[i] = ultrasound intensity at that point

    We first accumulate the samples on a regular voxel grid and then
    apply the same Gaussian blur to:
        1. intensity * sample_weight
        2. sample_weight

    The final voxel intensity is:

        I(x) = Gaussian(I * W) / Gaussian(W)

    This is normalized convolution. It prevents empty voxels from
    becoming artificially dark merely because they received fewer
    samples.

    IMPORTANT:
    floor() is used ONLY to place raw samples into the initial grid.
    It is no longer the final reconstruction rule. Gaussian filtering
    spreads each sample over a spatial neighborhood.

    sigma_mm:
        Gaussian standard deviation in physical units (mm).
        Larger values fill larger gaps but blur more detail.

    truncate:
        Kernel radius in units of sigma. For example, 3 means the
        Gaussian extends approximately +/- 3 sigma.

    min_weight:
        Voxels whose normalized Gaussian support is below this value
        are treated as unobserved and removed from the displayed cloud.
    """
    if len(points) == 0:
        return np.empty((0, 3)), np.empty((0,))

    voxel_size = float(voxel_size)
    sigma_mm = float(sigma_mm)

    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    if sigma_mm <= 0:
        raise ValueError("sigma_mm must be > 0")

    # Convert physical coordinates to integer voxel coordinates.
    # This is only the initial accumulation step; Gaussian filtering
    # below spreads the samples continuously over neighboring voxels.
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)

    # Shift the voxel coordinates so that the allocated numpy volume
    # starts at index 0 in each direction. This also supports negative
    # world coordinates without allocating unnecessary empty space.
    min_idx = voxel_indices.min(axis=0)
    max_idx = voxel_indices.max(axis=0)
    shape = tuple((max_idx - min_idx + 1).tolist())

    local_indices = voxel_indices - min_idx

    flat_indices = np.ravel_multi_index(
        local_indices.T,
        dims=shape,
    )

    # Accumulate both intensity and sample support.
    intensity_sum = np.bincount(
        flat_indices,
        weights=intensities.astype(np.float64),
        minlength=int(np.prod(shape)),
    ).reshape(shape)

    sample_weight = np.bincount(
        flat_indices,
        weights=np.ones(len(points), dtype=np.float64),
        minlength=int(np.prod(shape)),
    ).reshape(shape)

    # Convert the physical Gaussian bandwidth to voxel units.
    sigma_vox = sigma_mm / voxel_size

    from scipy.ndimage import gaussian_filter

    blurred_intensity = gaussian_filter(
        intensity_sum,
        sigma=sigma_vox,
        mode="constant",
        cval=0.0,
        truncate=truncate,
    )

    blurred_weight = gaussian_filter(
        sample_weight,
        sigma=sigma_vox,
        mode="constant",
        cval=0.0,
        truncate=truncate,
    )

    # Normalized convolution:
    # intensity estimate = weighted intensity / total weight
    valid = blurred_weight > float(min_weight)

    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0,))

    reconstructed_intensity = np.zeros_like(blurred_intensity)
    reconstructed_intensity[valid] = (
        blurred_intensity[valid] / blurred_weight[valid]
    )

    # Convert valid voxel indices back to physical voxel-center coordinates.
    valid_indices = np.argwhere(valid)
    global_indices = valid_indices + min_idx

    voxel_centers = (
        global_indices.astype(np.float64) + 0.5
    ) * voxel_size

    voxel_values = reconstructed_intensity[valid]

    return voxel_centers, voxel_values
