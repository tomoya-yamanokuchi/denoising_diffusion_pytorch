python - <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np

ep = Path("/home/dev/workspace/dataset/nedo_dismantling_log/eval/unet_D64_T1000_S20_simple_2d_20260605_133339/simple_paper_A_T8_N6_eta0p5_D2_w0p2_M32_S20_E100000_proposed_A_safety_margin_analysis/epsilon_greedy_00/Object_A/episode_4")

paths = [
    ep / "0_seq_obs_cast_z_axis0_0.png",
    ep / "conditions/seq_obs_cast_0_axis_z_0.png",
    ep / "inference_conditions/step_0/normalized_cond_used.png",
    ep / "inference_conditions/step_0/mask_observed.png",
    ep / "raw_pred_image/step_0/ensemble_z_0.png",
]

for p in paths:
    print("\n", p)
    if not p.exists():
        print("  MISSING")
        continue
    arr = np.asarray(Image.open(p).convert("RGB"))
    print("  shape:", arr.shape)
    print("  min/max:", arr.min(), arr.max())
    print("  nonzero pixels:", np.count_nonzero(arr))
    print("  unique colors:", len(np.unique(arr.reshape(-1, 3), axis=0)))
PY
