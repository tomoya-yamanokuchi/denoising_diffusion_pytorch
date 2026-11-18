# scripts/analysis/compare_case_predictions.py
"""
評価ログから、複数のケース (配置) の「観測 (モデルへの入力)」と「予測画像」を
計画ステップごとに並べた図を作る。予測がどの段階で分かれるかを見比べる用。

各行 = 計画ステップ k、各ケースにつき次の列を並べる:
  - Observation   : conditions/seq_obs_cast_{k}_axis_z_0.png (そのステップでモデルに入力された観測)
  - Prediction    : raw_pred_image/step_{k}/ensemble_z_*.png の平均 (M 枚の予測の平均)
  - GT target     : k_cost_map_logs.pickle の正解の目標部品マスク (記録があり、画像と同じ形のとき)
行の見出しには、選ばれた切断 (slice_range) と、目標部品がある面を「安全」と誤判定した数
(false_safe_count、記録があるとき) を書く。

使い方 (リポジトリのルートで):
  python scripts/analysis/compare_case_predictions.py \
    --root <評価結果のフォルダ (…/epsilon_greedy_00)> \
    --cases objF_type2_sample_001,objF_type2_sample_009 \
    --episode 0 \
    --out <出力先>/compare_vaeac_F.png
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def mean_prediction(step_dir: Path) -> tuple[np.ndarray | None, int]:
    files = sorted(step_dir.glob("ensemble_z_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        return None, 0
    return np.mean([load_rgb(f) for f in files], axis=0), len(files)


def load_log(ep_dir: Path, k: int) -> dict:
    path = ep_dir / f"{k}_cost_map_logs.pickle"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return pickle.load(f)


def gt_mask(log: dict, shape: tuple[int, int]) -> np.ndarray | None:
    try:
        m = np.asarray(log["target_soft_iou"]["ground_truth_mask"], dtype=float)
    except (KeyError, TypeError):
        return None
    return m if m.shape == shape else None


def step_title(k: int, log: dict) -> str:
    parts = [f"step {k}"]
    if "slice_range" in log:
        parts.append(f"cut={log['slice_range']}")
    fs = log.get("false_safe_rate")
    if isinstance(fs, dict) and "false_safe_count" in fs:
        parts.append(f"false-safe={fs['false_safe_count']}")
    return "  ".join(parts)


def available_steps(ep_dir: Path) -> list[int]:
    ks = set()
    for p in (ep_dir / "conditions").glob("seq_obs_cast_*_axis_z_0.png"):
        m = re.match(r"seq_obs_cast_(\d+)_axis_z_0\.png", p.name)
        if m:
            ks.add(int(m.group(1)))
    for p in ep_dir.glob("*_cost_map_logs.pickle"):
        ks.add(int(p.name.split("_")[0]))
    return sorted(ks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="評価結果のフォルダ (…/epsilon_greedy_00 など)")
    p.add_argument("--cases", type=str, required=True, help="比べるケース名 (カンマ区切り, 2〜3 個)")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--steps", type=str, default=None, help="表示するステップ (例: 0,1,2)。省略時は全ステップ")
    p.add_argument("--out", type=Path, required=True, help="出力する画像 (.png / .pdf)")
    args = p.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    eps = {c: args.root / c / f"episode_{args.episode}" for c in cases}
    for c, d in eps.items():
        if not d.is_dir():
            raise SystemExit(f"フォルダがありません: {d}")

    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    else:
        steps = sorted(set().union(*[available_steps(d) for d in eps.values()]))
    if not steps:
        raise SystemExit("表示できるステップがありません (conditions/ や *_cost_map_logs.pickle が見つからない)")

    # ケースごとの列: Observation, Prediction, (GT target があれば)
    has_gt = {}
    for c, d in eps.items():
        log = load_log(d, steps[0])
        pred, _ = mean_prediction(d / "raw_pred_image" / f"step_{steps[0]}")
        has_gt[c] = pred is not None and gt_mask(log, pred.shape[:2]) is not None
    col_spec = [(c, kind) for c in cases for kind in (["obs", "pred", "gt"] if has_gt[c] else ["obs", "pred"])]

    fig, axes = plt.subplots(len(steps), len(col_spec), figsize=(2.2 * len(col_spec), 2.3 * len(steps)),
                             squeeze=False)
    for i, k in enumerate(steps):
        for j, (c, kind) in enumerate(col_spec):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            d = eps[c]
            log = load_log(d, k)
            if kind == "obs":
                img = load_rgb(d / "conditions" / f"seq_obs_cast_{k}_axis_z_0.png")
                label = "Observation"
            elif kind == "pred":
                img, n = mean_prediction(d / "raw_pred_image" / f"step_{k}")
                label = f"Prediction (mean of {n})"
            else:
                pred, _ = mean_prediction(d / "raw_pred_image" / f"step_{k}")
                img = gt_mask(log, pred.shape[:2]) if pred is not None else None
                label = "GT target"
            if img is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="nearest")
            if i == 0:
                ax.set_title(f"{c}\n{label}", fontsize=7)
            if kind == "obs":
                ax.set_ylabel(step_title(k, log), fontsize=7)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"[OK] {args.out}  (cases={cases}, episode={args.episode}, steps={steps})")


if __name__ == "__main__":
    main()
