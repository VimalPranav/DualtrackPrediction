import numpy as np

def bin_fill(points, intensities, voxel_size):
    """
    Pixel-based bin filling.

    Each 3D sample is assigned to one voxel. If multiple samples hit
    the same voxel, their intensity values are averaged.

    Returns voxel centers and their averaged ultrasound intensity.
    """
    if len(points) == 0:
        return np.empty((0, 3)), np.empty((0,))

    voxel_indices = np.floor(
        points / float(voxel_size)
    ).astype(np.int64)

    unique_voxels, inverse = np.unique(
        voxel_indices,
        axis=0,
        return_inverse=True,
    )

    sums = np.bincount(inverse, weights=intensities)
    counts = np.bincount(inverse)

    voxel_intensities = sums / np.maximum(counts, 1)
    voxel_centers = (
        unique_voxels.astype(np.float64) + 0.5
    ) * float(voxel_size)

    return voxel_centers, voxel_intensities