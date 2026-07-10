import h5py
import numpy as np

f = h5py.File(
    "/home/user/Desktop/ULTRASOUND/Dataset/000/LH_Par_C_DtP.h5",
    "r"
)

T = f["tracking"][0]

print(T)

R = T[:3, :3]

print("Determinant:", np.linalg.det(R))