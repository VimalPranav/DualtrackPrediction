"""
visualize_dualtrack.py - Run from DualTrack repo root.

Usage:
    python visualize_dualtrack.py \
        --config configs/dualtrack_evaluation/dualtrack_final_local.yaml \
        --checkpoint checkpoints/dualtrack_final.pt \
        --out_dir results/visuals/
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

C_GT="#00C2FF"; C_PRED="#FF6B35"; C_ERR="#FF3B5C"
C_BG="#0D1117"; C_GRID="#21262D"; C_TEXT="#E6EDF3"

def sixdof_to_matrix(poses):
    """Convert (N, 6) -> (N, 4, 4). poses = [tx,ty,tz, rx,ry,rz] in mm/radians."""
    N = len(poses)
    mats = np.zeros((N, 4, 4))
    mats[:, 3, 3] = 1.0
    mats[:, :3, 3] = poses[:, :3]
    mats[:, :3, :3] = Rotation.from_euler("xyz", poses[:, 3:], degrees=True).as_matrix()
    return mats

def compose_relative_to_absolute(rel_mats):
    """(N, 4, 4) relative -> (N+1, 4, 4) absolute starting from identity."""
    N = len(rel_mats)
    abs_mats = np.zeros((N+1, 4, 4))
    abs_mats[0] = np.eye(4)
    for i in range(N):
        abs_mats[i+1] = abs_mats[i] @ rel_mats[i]
    return abs_mats

def compute_errors(pred, gt, spacing):
    """Mean corner point error in mm between predicted and GT transforms."""
    H, W = 480, 640   # default image size for TUS-REC
    sx, sy = float(spacing[0]), float(spacing[1])
    corners = np.array([
        [-W/2*sx, -H/2*sy, 0, 1], [ W/2*sx, -H/2*sy, 0, 1],
        [-W/2*sx,  H/2*sy, 0, 1], [ W/2*sx,  H/2*sy, 0, 1]
    ]).T
    N = min(len(pred), len(gt))
    return np.array([
        np.mean(np.linalg.norm((pred[i]@corners)[:3] - (gt[i]@corners)[:3], axis=0))
        for i in range(N)
    ])

def dark_ax(ax):
    ax.set_facecolor(C_BG); ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.xaxis.label.set_color(C_TEXT); ax.yaxis.label.set_color(C_TEXT)
    for s in ax.spines.values(): s.set_edgecolor(C_GRID)
    ax.grid(color=C_GRID, linewidth=0.5); return ax

def plot_trajectory(pred, gt, path):
    pp = (pred[:, :3, 3] - gt[0, :3, 3]) / 10.0
    gp = (gt[:, :3, 3]  - gt[0, :3, 3]) / 10.0
    fig = plt.figure(figsize=(10, 8)); fig.patch.set_facecolor(C_BG)
    ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor(C_BG)
    ax.plot(*gp.T, color=C_GT,   lw=2, label="Ground Truth")
    ax.plot(*pp.T, color=C_PRED, lw=2, linestyle="--", label="DualTrack Pred")
    step = max(1, len(pp)//20)
    for i in range(0, len(pp), step):
        ax.plot([pp[i,0],gp[i,0]], [pp[i,1],gp[i,1]], [pp[i,2],gp[i,2]],
                color=C_ERR, lw=0.6, alpha=0.5)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(C_GRID); ax.yaxis.pane.set_edgecolor(C_GRID); ax.zaxis.pane.set_edgecolor(C_GRID)
    ax.set_xlabel("X (cm)", color=C_TEXT); ax.set_ylabel("Y (cm)", color=C_TEXT); ax.set_zlabel("Z (cm)", color=C_TEXT)
    ax.set_title("3D Probe Trajectory (1:1 Physical Scale)", color=C_TEXT, fontsize=14)
    
    # Enforce 1:1:1 aspect ratio so the plot isn't stretched
    x_range = np.ptp(gp[:, 0])
    y_range = np.ptp(gp[:, 1])
    z_range = np.ptp(gp[:, 2])
    max_range = max(x_range, y_range, z_range)
    
    mid_x = (np.max(gp[:, 0]) + np.min(gp[:, 0])) * 0.5
    mid_y = (np.max(gp[:, 1]) + np.min(gp[:, 1])) * 0.5
    mid_z = (np.max(gp[:, 2]) + np.min(gp[:, 2])) * 0.5
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    ax.set_box_aspect((1, 1, 1))

    ax.legend(facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG); plt.close()
    print(f"  ✔  Trajectory   → {path}")

def plot_trajectory_interactive(pred, gt, path):
    import plotly.graph_objects as go
    
    pp = (pred[:, :3, 3] - gt[0, :3, 3]) / 10.0
    gp = (gt[:, :3, 3]  - gt[0, :3, 3]) / 10.0
    
    fig = go.Figure()
    
    # Ground Truth
    fig.add_trace(go.Scatter3d(
        x=gp[:, 0], y=gp[:, 1], z=gp[:, 2],
        mode='lines', name='Ground Truth',
        line=dict(color='#00B4D8', width=5)
    ))
    
    # Prediction
    fig.add_trace(go.Scatter3d(
        x=pp[:, 0], y=pp[:, 1], z=pp[:, 2],
        mode='lines', name='DualTrack Pred',
        line=dict(color='#FF7B00', width=5, dash='dash')
    ))
    
    # Error lines (subsampled)
    step = max(1, len(pp)//20)
    for i in range(0, len(pp), step):
        fig.add_trace(go.Scatter3d(
            x=[pp[i,0], gp[i,0]], y=[pp[i,1], gp[i,1]], z=[pp[i,2], gp[i,2]],
            mode='lines', showlegend=False,
            line=dict(color='#E63946', width=2)
        ))
        
    fig.update_layout(
        title="Interactive 3D Probe Trajectory (cm)",
        scene=dict(
            xaxis_title='X (cm)',
            yaxis_title='Y (cm)',
            zaxis_title='Z (cm)',
            aspectmode='data'  # Enforces 1:1:1 true physical scaling
        ),
        paper_bgcolor='rgba(15,15,15,1)',
        plot_bgcolor='rgba(15,15,15,1)',
        font=dict(color='white')
    )
    
    fig.write_html(path)
    print(f"  ✔  Interactive(cm)→ {path}")
    
    # Automatically open the generated HTML file in the default browser
    import webbrowser
    import os
    webbrowser.open('file://' + os.path.realpath(path))

def plot_trajectory_interactive_mm(pred, gt, path):
    import plotly.graph_objects as go
    
    pp = pred[:, :3, 3] - gt[0, :3, 3]
    gp = gt[:, :3, 3]  - gt[0, :3, 3]
    
    fig = go.Figure()
    
    # Ground Truth
    fig.add_trace(go.Scatter3d(
        x=gp[:, 0], y=gp[:, 1], z=gp[:, 2],
        mode='lines', name='Ground Truth',
        line=dict(color='#00B4D8', width=5)
    ))
    
    # Prediction
    fig.add_trace(go.Scatter3d(
        x=pp[:, 0], y=pp[:, 1], z=pp[:, 2],
        mode='lines', name='DualTrack Pred',
        line=dict(color='#FF7B00', width=5, dash='dash')
    ))
    
    # Error lines (subsampled)
    step = max(1, len(pp)//20)
    for i in range(0, len(pp), step):
        fig.add_trace(go.Scatter3d(
            x=[pp[i,0], gp[i,0]], y=[pp[i,1], gp[i,1]], z=[pp[i,2], gp[i,2]],
            mode='lines', showlegend=False,
            line=dict(color='#E63946', width=2)
        ))
        
    fig.update_layout(
        title="Interactive 3D Probe Trajectory (mm)",
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'  # Enforces 1:1:1 true physical scaling
        ),
        paper_bgcolor='rgba(15,15,15,1)',
        plot_bgcolor='rgba(15,15,15,1)',
        font=dict(color='white')
    )
    
    fig.write_html(path)
    print(f"  ✔  Interactive(mm)→ {path}")
    # Don't auto-open both to avoid spamming the browser

def plot_error_curve(errors, path):
    fig, ax = plt.subplots(figsize=(12, 4)); fig.patch.set_facecolor(C_BG); dark_ax(ax)
    t = np.arange(len(errors))
    ax.fill_between(t, 0, errors, color=C_ERR, alpha=0.25)
    ax.plot(t, errors, color=C_ERR, lw=1.5)
    ax.axhline(errors.mean(), color="white", lw=1, linestyle="--", label=f"Mean: {errors.mean():.2f} mm")
    ax.axhline(np.percentile(errors,95), color=C_GT, lw=0.8, linestyle=":", label=f"P95: {np.percentile(errors,95):.2f} mm")
    ax.set_xlabel("Frame index", color=C_TEXT); ax.set_ylabel("GPE (mm)", color=C_TEXT)
    ax.set_title("Per-frame Global Position Error", color=C_TEXT, fontsize=13)
    ax.legend(facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG); plt.close()
    print(f"  ✔  Error curve  → {path}")

def plot_frames(images, errors, path, n=6):
    N = len(images)
    idxs = np.linspace(0, N-1, n, dtype=int)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6)); fig.patch.set_facecolor(C_BG)
    for col, i in enumerate(idxs):
        axes[0,col].imshow(images[i], cmap="gray", vmin=0, vmax=255, aspect="auto")
        axes[0,col].set_title(f"Frame {i}", color=C_TEXT, fontsize=9); axes[0,col].axis("off")
        ax = axes[1,col]; ax.set_facecolor(C_BG)
        c = C_ERR if errors[i] > errors.mean() else C_GT
        ax.bar([0], [errors[i]], color=c, width=0.6)
        ax.set_ylim(0, errors.max()*1.15); ax.set_xlim(-0.5, 0.5); ax.set_xticks([])
        ax.set_title(f"{errors[i]:.1f} mm", color=c, fontsize=9)
        ax.tick_params(colors=C_TEXT, labelsize=7)
        for s in ax.spines.values(): s.set_edgecolor(C_GRID)
        ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    fig.suptitle("Sample Frames + Per-frame GPE", color=C_TEXT, fontsize=13)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG); plt.close()
    print(f"  ✔  Sample frames→ {path}")

def plot_dashboard(pred, gt, errors, images, path):
    pp = pred[:,:3,3] - gt[0,:3,3]
    gp = gt[:,:3,3]  - gt[0,:3,3]
    fig = plt.figure(figsize=(18, 10)); fig.patch.set_facecolor(C_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax3 = fig.add_subplot(gs[0,0], projection="3d"); ax3.set_facecolor(C_BG)
    ax3.plot(*gp.T, color=C_GT, lw=1.8, label="GT")
    ax3.plot(*pp.T, color=C_PRED, lw=1.8, linestyle="--", label="Pred")
    ax3.tick_params(colors=C_TEXT, labelsize=7)
    ax3.xaxis.pane.fill = ax3.yaxis.pane.fill = ax3.zaxis.pane.fill = False
    ax3.set_title("3D Trajectory", color=C_TEXT, fontsize=11)
    ax3.legend(fontsize=8, facecolor=C_BG, labelcolor=C_TEXT, edgecolor=C_GRID)

    ax_xy = fig.add_subplot(gs[0,1]); dark_ax(ax_xy)
    ax_xy.plot(gp[:,0], gp[:,1], color=C_GT, lw=1.8, label="GT")
    ax_xy.plot(pp[:,0], pp[:,1], color=C_PRED, lw=1.8, linestyle="--", label="Pred")
    ax_xy.set_xlabel("X (mm)"); ax_xy.set_ylabel("Y (mm)")
    ax_xy.set_title("XY Projection", color=C_TEXT, fontsize=11)
    ax_xy.legend(fontsize=8, facecolor=C_BG, labelcolor=C_TEXT, edgecolor=C_GRID)

    ax_xz = fig.add_subplot(gs[0,2]); dark_ax(ax_xz)
    ax_xz.plot(gp[:,0], gp[:,2], color=C_GT, lw=1.8, label="GT")
    ax_xz.plot(pp[:,0], pp[:,2], color=C_PRED, lw=1.8, linestyle="--", label="Pred")
    ax_xz.set_xlabel("X (mm)"); ax_xz.set_ylabel("Z (mm)")
    ax_xz.set_title("XZ Projection", color=C_TEXT, fontsize=11)
    ax_xz.legend(fontsize=8, facecolor=C_BG, labelcolor=C_TEXT, edgecolor=C_GRID)

    ax_e = fig.add_subplot(gs[1,:2]); dark_ax(ax_e)
    t = np.arange(len(errors))
    ax_e.fill_between(t, 0, errors, color=C_ERR, alpha=0.2)
    ax_e.plot(t, errors, color=C_ERR, lw=1.5)
    ax_e.axhline(errors.mean(), color="white", lw=1, linestyle="--", label=f"Mean: {errors.mean():.2f} mm")
    ax_e.axhline(np.percentile(errors,95), color=C_GT, lw=0.8, linestyle=":", label=f"P95: {np.percentile(errors,95):.2f} mm")
    ax_e.set_xlabel("Frame"); ax_e.set_ylabel("GPE (mm)")
    ax_e.set_title("Per-frame Error", color=C_TEXT, fontsize=11)
    ax_e.legend(fontsize=8, facecolor=C_BG, labelcolor=C_TEXT, edgecolor=C_GRID)

    ax_t = fig.add_subplot(gs[1,2]); ax_t.set_facecolor(C_BG); ax_t.axis("off")
    stats = [["Metric","Value"],
             ["Frames",     f"{len(errors)}"],
             ["Mean GPE",   f"{errors.mean():.2f} mm"],
             ["Median GPE", f"{np.median(errors):.2f} mm"],
             ["P95 GPE",    f"{np.percentile(errors,95):.2f} mm"],
             ["Max GPE",    f"{errors.max():.2f} mm"],
             ["Min GPE",    f"{errors.min():.2f} mm"],
             ["Std GPE",    f"{errors.std():.2f} mm"]]
    tbl = ax_t.table(cellText=stats[1:], colLabels=stats[0], cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_facecolor("#161B22" if r%2==0 else C_BG)
        cell.set_edgecolor(C_GRID); cell.set_text_props(color=C_TEXT)
    ax_t.set_title("Summary Statistics", color=C_TEXT, fontsize=11)
    fig.suptitle("DualTrack — Inference Results", color=C_TEXT, fontsize=16, fontweight="bold")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG); plt.close()
    print(f"  ✔  Dashboard    → {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir",    default="results/visuals")
    parser.add_argument("--n_sweeps",   type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'='*55}\n  DualTrack Visual Inference\n{'='*55}")

    # load config + data
    cfg = OmegaConf.load(args.config)
    cfg.device = "cpu"; cfg.use_amp = False
    cfg.data.limit_scans = args.n_sweeps
    cfg.data.num_workers = 0

    from src.datasets import get_dataloaders
    from src.models import get_model

    print("[1/3] Loading data and model...")
    _, val_loader = get_dataloaders(**cfg.data)

    model_cfg = OmegaConf.load("configs/model/dualtrack.yaml")
    if "model" in cfg and "local_encoder_cfg" in cfg.model:
        model_cfg = cfg.model
    model_cfg.checkpoint = args.checkpoint
    model = get_model(**model_cfg); model.eval()
    print("      Ready.")

    print(f"[2/3] Running inference on {args.n_sweeps} sweep(s)...")
    results = []
    for i, batch in enumerate(val_loader):
        if i >= args.n_sweeps:
            break
        sweep_id = batch["sweep_id"][0]
        print(f"      [{i+1}/{args.n_sweeps}] {sweep_id}")

        with torch.no_grad():
            g = batch["global_encoder_images"]
            l = batch["local_encoder_images"]
            out = model(g, l)  # (1, N-1, 6)

        # predicted 6-DOF -> 4x4 absolute
        pred_6dof = out.squeeze(0).cpu().numpy()          # (N-1, 6)
        
        # Apply Causal EMA to relative 6DOF predictions (Real-time compatible)
        alpha = 0.5 # 1.0 is no smoothing, lower is more smoothing
        pred_6dof_ema = np.zeros_like(pred_6dof)
        pred_6dof_ema[0] = pred_6dof[0]
        for i in range(1, len(pred_6dof)):
            pred_6dof_ema[i] = alpha * pred_6dof[i] + (1 - alpha) * pred_6dof_ema[i-1]
            
        pred_rel  = sixdof_to_matrix(pred_6dof_ema)           # (N-1, 4, 4)
        pred_abs  = compose_relative_to_absolute(pred_rel) # (N, 4, 4)

        gt = batch["tracking"][0]   # (N, 4, 4) numpy
        if hasattr(gt, "numpy"): gt = gt.numpy()
        gt = np.linalg.inv(gt[0]) @ gt
        spacing = batch["spacing"]
        if hasattr(spacing, "numpy"):
            spacing = spacing.squeeze(0).numpy()
        elif isinstance(spacing, list):
            spacing = np.array(spacing[0])

        # raw images for plotting
        imgs = batch["local_encoder_images"].squeeze(0).cpu().numpy()  # (N, 1, H, W)
        imgs = imgs[:, 0]  # (N, H, W)
        imgs = (imgs * 255).clip(0, 255).astype(np.uint8)

        results.append((sweep_id, pred_abs, gt, imgs, spacing))

    print("[3/3] Generating visuals...")
    for sweep_id, pred_abs, gt, imgs, spacing in results:
        errors = compute_errors(pred_abs, gt, spacing)
        print(f"\n  {sweep_id}  |  Mean GPE: {errors.mean():.2f} mm  |  Max: {errors.max():.2f} mm")

        out_dir = os.path.join(args.out_dir, sweep_id)
        os.makedirs(out_dir, exist_ok=True)

        N = min(len(pred_abs), len(gt))
        plot_trajectory(pred_abs[:N], gt[:N],         os.path.join(out_dir, "trajectory_3d.png"))
        plot_trajectory_interactive(pred_abs[:N], gt[:N], os.path.join(out_dir, "trajectory_3d_interactive.html"))
        plot_trajectory_interactive_mm(pred_abs[:N], gt[:N], os.path.join(out_dir, "trajectory_3d_interactive_mm.html"))
        plot_error_curve(errors,                      os.path.join(out_dir, "error_curve.png"))
        plot_frames(imgs[:N], errors,                 os.path.join(out_dir, "sample_frames.png"))
        plot_dashboard(pred_abs[:N], gt[:N], errors, imgs[:N], os.path.join(out_dir, "dashboard.png"))

    print(f"\n{'='*55}")
    print(f"  Done! Results in: {os.path.abspath(args.out_dir)}/")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()