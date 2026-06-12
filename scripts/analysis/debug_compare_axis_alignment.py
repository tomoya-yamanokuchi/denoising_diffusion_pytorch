from pathlib import Path
import numpy as np

# debug_dir = Path("analysis/debug_axis_alignment/Object_A_ep0_step0")
# debug_dir = Path("analysis/debug_axis_alignment/Object_B_ep0_step0")
debug_dir = Path("analysis/debug_axis_alignment/Object_C_ep0_step0")

presence = np.load(debug_dir / "presence_volume/presence_score_r0.npy")
axis = np.load(debug_dir / "axis_arrays/decision_axis_cost_r0.npz")

p = {
    "x": presence.max(axis=(1, 2)),
    "y": presence.max(axis=(0, 2)),
    "z": presence.max(axis=(0, 1)),
}

u = {
    "x": axis["ucb_x"],
    "y": axis["ucb_y"],
    "z": axis["ucb_z"],
}

def normalize(a):
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.max() > a.min():
        return (a - a.min()) / (a.max() - a.min())
    return a * 0.0

def mse(a, b):
    a = normalize(a)
    b = normalize(b)
    return float(np.mean((a - b) ** 2))

print("=== shapes and peaks ===")
for name, arr in p.items():
    arr = np.asarray(arr).reshape(-1)
    print(f"presence_{name}: shape={arr.shape}, argmax={arr.argmax()}, max={arr.max():.3f}, values={np.round(arr, 3)}")
for name, arr in u.items():
    arr = np.asarray(arr).reshape(-1)
    print(f"ucb_{name}:      shape={arr.shape}, argmax={arr.argmax()}, max={arr.max():.3f}, values={np.round(arr, 3)}")

print("\n=== pairwise MSE: lower is better ===")
for u_name, u_arr in u.items():
    for p_name, p_arr in p.items():
        same = mse(u_arr, p_arr)
        flip = mse(u_arr, p_arr[::-1])
        best = "same" if same <= flip else "flip"
        print(
            f"ucb_{u_name} vs presence_{p_name}: "
            f"same={same:.4f}, flip={flip:.4f}, best={best}"
        )
