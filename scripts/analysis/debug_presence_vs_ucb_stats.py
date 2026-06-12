from pathlib import Path
import pickle
import numpy as np

from denoising_diffusion_pytorch.cost.types import AxisCost


COST_LOG = Path(
    "/home/dev/workspace/dataset/nedo_dismantling_log/eval/"
    "unet_D64_T1000_S20_simple_2d_20260605_133339/"
    "simple_paper_A_T8_N6_eta0p5_D0_w0p2_M32_S20_E100000_proposed_A/"
    "epsilon_greedy_00/Object_A/episode_0/0_cost_map_logs.pickle"
)
DEBUG_DIR = Path("analysis/debug_axis_alignment/Object_A_ep0_step0")
TARGET = "blue"


def remap_to_presence_coords(axis_cost: AxisCost) -> AxisCost:
    """
    Debug result from Object A/B/C:
      clip x_axis -> presence y_axis
      clip y_axis -> presence x_axis reversed
      clip z_axis -> presence z_axis
    """
    return AxisCost(
        x_axis=np.asarray(axis_cost.y_axis, dtype=float).reshape(-1)[::-1],
        y_axis=np.asarray(axis_cost.x_axis, dtype=float).reshape(-1),
        z_axis=np.asarray(axis_cost.z_axis, dtype=float).reshape(-1),
    )


def summarize_array(name, arr, thresholds=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5)):
    arr = np.asarray(arr, dtype=float)
    flat = arr.reshape(-1)

    print(f"\n=== {name} ===")
    print(f"shape      : {arr.shape}")
    print(f"min / max  : {flat.min():.6f} / {flat.max():.6f}")
    print(f"mean / std : {flat.mean():.6f} / {flat.std():.6f}")
    print(f"nonzero    : {(flat > 0).sum()} / {flat.size}")

    positive = flat[flat > 0]
    if positive.size > 0:
        print(f"positive min / median / max: {positive.min():.6f} / {np.median(positive):.6f} / {positive.max():.6f}")

    for th in thresholds:
        print(f"count > {th:>4}: {(flat > th).sum():4d} / {flat.size}")


def normalize(a):
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.max() > a.min():
        return (a - a.min()) / (a.max() - a.min())
    return a * 0.0


def summarize_profile(name, arr):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    print(f"{name}: argmax={arr.argmax()}, max={arr.max():.6f}")
    print(f"  {np.round(arr, 4)}")


# ---------------------------------------------------------------------
# 1. Load voxel-level presence frequency volume.
# ---------------------------------------------------------------------
presence = np.load(DEBUG_DIR / "presence_volume/presence_score_r0.npy")

# Voxel-level mean presence.
p_x = presence.max(axis=(1, 2))
p_y = presence.max(axis=(0, 2))
p_z = presence.max(axis=(0, 1))

# Since presence is an ensemble average of binary masks, a voxel-level
# Bernoulli std can be estimated as sqrt(p * (1 - p)).
voxel_std = np.sqrt(presence * (1.0 - presence))
std_x = voxel_std.max(axis=(1, 2))
std_y = voxel_std.max(axis=(0, 2))
std_z = voxel_std.max(axis=(0, 1))

print("\n############################")
print("# Presence frequency volume")
print("############################")
summarize_array("presence volume p(x,y,z)", presence)
summarize_array("voxel Bernoulli std sqrt(p(1-p))", voxel_std)

print("\n--- presence max-projection profiles ---")
summarize_profile("presence_x = max_yz p", p_x)
summarize_profile("presence_y = max_xz p", p_y)
summarize_profile("presence_z = max_xy p", p_z)

print("\n--- voxel std max-projection profiles ---")
summarize_profile("voxel_std_x", std_x)
summarize_profile("voxel_std_y", std_y)
summarize_profile("voxel_std_z", std_z)


# ---------------------------------------------------------------------
# 2. Load cost_ensembles and compute mean/std/UCB before display.
# ---------------------------------------------------------------------
with COST_LOG.open("rb") as f:
    logs = pickle.load(f)

cost_ensemble = getattr(logs["cost_ensembles"], TARGET)

raw_bool = AxisCost(
    x_axis=(np.asarray(cost_ensemble.x_axis) > 0).astype(float),
    y_axis=(np.asarray(cost_ensemble.y_axis) > 0).astype(float),
    z_axis=(np.asarray(cost_ensemble.z_axis) > 0).astype(float),
)

mean_raw = AxisCost(
    x_axis=raw_bool.x_axis.mean(axis=0),
    y_axis=raw_bool.y_axis.mean(axis=0),
    z_axis=raw_bool.z_axis.mean(axis=0),
)

std_raw = AxisCost(
    x_axis=raw_bool.x_axis.std(axis=0),
    y_axis=raw_bool.y_axis.std(axis=0),
    z_axis=raw_bool.z_axis.std(axis=0),
)

ucb_raw = AxisCost(
    x_axis=mean_raw.x_axis + std_raw.x_axis,
    y_axis=mean_raw.y_axis + std_raw.y_axis,
    z_axis=mean_raw.z_axis + std_raw.z_axis,
)

# Convert to presence-coordinate system for comparison.
mean_disp = remap_to_presence_coords(mean_raw)
std_disp = remap_to_presence_coords(std_raw)
ucb_disp = remap_to_presence_coords(ucb_raw)

print("\n############################")
print("# Axis-wise cutting risk")
print("############################")

print("\n--- mean profiles, remapped to presence coords ---")
summarize_profile("mean_x_display", mean_disp.x_axis)
summarize_profile("mean_y_display", mean_disp.y_axis)
summarize_profile("mean_z_display", mean_disp.z_axis)

print("\n--- std profiles, remapped to presence coords ---")
summarize_profile("std_x_display", std_disp.x_axis)
summarize_profile("std_y_display", std_disp.y_axis)
summarize_profile("std_z_display", std_disp.z_axis)

print("\n--- UCB = mean + std profiles, remapped to presence coords ---")
summarize_profile("ucb_x_display", ucb_disp.x_axis)
summarize_profile("ucb_y_display", ucb_disp.y_axis)
summarize_profile("ucb_z_display", ucb_disp.z_axis)


# ---------------------------------------------------------------------
# 3. Compare presence projection vs mean/std/UCB.
# ---------------------------------------------------------------------
print("\n############################")
print("# Comparison")
print("############################")

pairs = [
    ("x", p_x, mean_disp.x_axis, std_disp.x_axis, ucb_disp.x_axis),
    ("y", p_y, mean_disp.y_axis, std_disp.y_axis, ucb_disp.y_axis),
    ("z", p_z, mean_disp.z_axis, std_disp.z_axis, ucb_disp.z_axis),
]

for axis_name, p, m, s, u in pairs:
    print(f"\n--- axis {axis_name} ---")
    print("idx | presence | mean | std | ucb")
    for i, (pv, mv, sv, uv) in enumerate(zip(p, m, s, u)):
        if pv > 0 or mv > 0 or sv > 0 or uv > 0:
            print(f"{i:2d} | {pv:8.4f} | {mv:6.4f} | {sv:6.4f} | {uv:6.4f}")

    print(
        f"nonzero counts: presence>0={np.sum(p > 0)}, "
        f"mean>0={np.sum(m > 0)}, std>0={np.sum(s > 0)}, ucb>0={np.sum(u > 0)}"
    )
    print(
        f"high counts: presence>0.5={np.sum(p > 0.5)}, "
        f"ucb>0.5={np.sum(u > 0.5)}, ucb>1.0={np.sum(u > 1.0)}"
    )
