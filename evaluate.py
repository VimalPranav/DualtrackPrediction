from omegaconf import OmegaConf 
from src.models import get_model 

cfg_path = '/home/user/Desktop/ULTRASOUND/DualtrackPrediction_vimal/config/model/dualtrack_ft_tus_rec_2025.yaml'
cfg = OmegaConf.load(cfg_path)
cfg.checkpoint = '/home/user/Desktop/ULTRASOUND/DualtrackPrediction_vimal/experiments/dualtrack_ft_tus_rec_2025_v3_best.pt'

model = get_model(**cfg)

import h5py

file_path = "/home/user/Desktop/ULTRASOUND/DualtrackPrediction_vimal/processed_data/000_LH_Par_C_DtP.h5"

with h5py.File(file_path, "r") as f:
    print("Datasets:", list(f.keys()))
    
    if "images" in f:
        print("Number of frames:", len(f["images"]))
    elif "frames" in f:
        print("Number of frames:", len(f["frames"]))