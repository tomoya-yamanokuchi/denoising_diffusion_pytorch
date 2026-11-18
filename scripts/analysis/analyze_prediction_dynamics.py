# scripts/analysis/analyze_prediction_dynamics.py
"""
評価ログから、予測の「多様性」と「観測に応じた更新」を手法ごとに定量化し、
配置 (configuration) を単位として手法間で比較する。

1. 予測の多様性 (各計画ステップの M 個のサンプルの間のばらつき)
   - pixel_disagreement : M 個のサンプルの間で、色の最大値と最小値の差がどれかのチャネルで
                          tol (既定 0.1 = 部品判定の色の許容幅) を超える画素の割合
   - surface_disagreement: 断面ごとの presence_probability (目標部品ありと予測したサンプルの割合)
                          が 0 でも 1 でもない (= サンプルの判断が割れている) 断面の割合
2. 観測に応じた予測の更新 (連続する計画ステップの間)
   - obs_change   : 観測画像 (モデルへの入力) の平均絶対変化
   - pred_change  : M 個の予測の平均画像の平均絶対変化
   - presence_change: presence_probability の平均絶対変化
   - pred_sensitivity: pred_change / obs_change (obs_change > --min-obs-change のステップのみ)
     観測がほとんど変わらないステップ (例: 安全な切断がなく切れなかった) は、予測が変わらない
     のが当然なので、感度の計算から除く

集計の単位
   同じエピソード内のステップは独立でないため、ステップ → エピソード → 配置 の順に平均し、
   配置を 1 単位とする。手法間の比較は、同じ配置どうしを対応させた Wilcoxon の符号順位検定。

使い方 (リポジトリのルートで):
  python scripts/analysis/analyze_prediction_dynamics.py \
    --roots VAEAC=<VAEAC の …_revision/epsilon_greedy_00 のフォルダ> ... \
            Proposed=<Proposed の …/epsilon_greedy_00 のフォルダ> ... \
    --out-dir <出力先>
  (同じラベルを複数回書けば、Object D/E/F のフォルダをまとめて1手法として扱う)
"""
from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from scipy.stats import wilcoxon
except Exception:  # scipy が無い・読み込めない環境では検定を省略する
    wilcoxon = None

STEP_LOG = re.compile(r"^(\d+)_cost_map_logs\.pickle$")


def load_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_samples(step_dir: Path) -> np.ndarray | None:
    files = sorted(step_dir.glob("ensemble_z_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        return None
    return np.stack([load_rgb(f) for f in files])          # [M, H, W, 3]


def presence(log: dict) -> np.ndarray | None:
    try:
        pp = log["brier_score"]["presence_probability"]
    except (KeyError, TypeError):
        return None
    return np.concatenate([np.asarray(pp[a], dtype=float).ravel() for a in ("x", "y", "z")])


def analyze_episode(ep: Path, tol: float) -> list[dict]:
    steps = sorted(int(STEP_LOG.match(p.name).group(1)) for p in ep.glob("*_cost_map_logs.pickle"))
    rows, prev = [], None
    for k in steps:
        with (ep / f"{k}_cost_map_logs.pickle").open("rb") as f:
            log = pickle.load(f)
        samples = load_samples(ep / "raw_pred_image" / f"step_{k}")
        obs = load_rgb(ep / "conditions" / f"seq_obs_cast_{k}_axis_z_0.png")
        p = presence(log)
        r = {"step": k}
        if samples is not None:
            spread = samples.max(axis=0) - samples.min(axis=0)           # [H, W, 3]
            r["pixel_disagreement"] = float((spread > tol).any(axis=-1).mean())
            mean_img = samples.mean(axis=0)
        else:
            mean_img = None
        if p is not None:
            r["surface_disagreement"] = float(((p > 0) & (p < 1)).mean())
        if prev is not None:
            if obs is not None and prev["obs"] is not None:
                r["obs_change"] = float(np.abs(obs - prev["obs"]).mean())
            if mean_img is not None and prev["mean_img"] is not None:
                r["pred_change"] = float(np.abs(mean_img - prev["mean_img"]).mean())
            if p is not None and prev["p"] is not None:
                r["presence_change"] = float(np.abs(p - prev["p"]).mean())
        rows.append(r)
        prev = {"obs": obs, "mean_img": mean_img, "p": p}
    return rows


def parse_roots(specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"--roots は Label=path の形で指定してください: {s}")
        label, path = s.split("=", 1)
        out.append((label.strip(), Path(path)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", required=True, help="Label=評価結果のフォルダ (複数可)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tol", type=float, default=0.1, help="画素の不一致とみなす色の差 (既定 0.1)")
    p.add_argument("--min-obs-change", type=float, default=1e-3,
                   help="感度の計算に使うステップの、観測の変化の下限")
    p.add_argument("--max-episodes", type=int, default=None, help="各配置で解析するエピソード数の上限")
    args = p.parse_args()

    step_rows = []
    for label, root in parse_roots(args.roots):
        cases = sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("episode_*")))
        print(f"[read] {label}: {root}  ({len(cases)} cases)", flush=True)
        for case in cases:
            eps = sorted(case.glob("episode_*"), key=lambda d: int(d.name.split("_")[1]))
            for ep in eps[: args.max_episodes]:
                for r in analyze_episode(ep, args.tol):
                    r.update(method=label, case=case.name, episode=ep.name)
                    step_rows.append(r)
    steps = pd.DataFrame(step_rows)
    if steps.empty:
        raise SystemExit("解析できるステップがありません")

    ok = steps["obs_change"] > args.min_obs_change if "obs_change" in steps else pd.Series(False, index=steps.index)
    steps["pred_sensitivity"] = np.where(ok, steps.get("pred_change") / steps.get("obs_change"), np.nan)

    metrics = [m for m in ["pixel_disagreement", "surface_disagreement", "obs_change", "pred_change",
                           "presence_change", "pred_sensitivity"] if m in steps]
    per_ep = steps.groupby(["method", "case", "episode"])[metrics].mean().reset_index()
    per_cfg = per_ep.groupby(["method", "case"])[metrics].mean().reset_index()
    steps["frozen_step"] = ~ok
    frozen = steps.dropna(subset=["obs_change"]).groupby("method")["frozen_step"].mean()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps.to_csv(args.out_dir / "prediction_dynamics_per_step.csv", index=False)
    per_cfg.to_csv(args.out_dir / "prediction_dynamics_per_configuration.csv", index=False)

    print("\n=== 配置ごとの平均の要約 (中央値 [25%, 75%]) ===")
    summ = []
    for m, g in per_cfg.groupby("method"):
        row = {"method": m, "configs": len(g)}
        for c in metrics:
            v = g[c].dropna()
            row[c] = (f"{v.median():.4f} [{v.quantile(.25):.4f}, {v.quantile(.75):.4f}]"
                      if len(v) else "n/a")
        row["frozen_step_ratio"] = f"{frozen.get(m, np.nan):.2f}"
        summ.append(row)
    summ = pd.DataFrame(summ)
    print(summ.to_string(index=False))
    summ.to_csv(args.out_dir / "prediction_dynamics_summary.csv", index=False)

    methods = list(dict.fromkeys(per_cfg["method"]))
    if wilcoxon is not None and len(methods) >= 2:
        print("\n=== 手法間の比較 (同じ配置どうしを対応させた Wilcoxon の符号順位検定, 両側) ===")
        tests = []
        base = methods[0]
        for other in methods[1:]:
            for c in metrics:
                a = per_cfg[per_cfg.method == base].set_index("case")[c]
                b = per_cfg[per_cfg.method == other].set_index("case")[c]
                pair = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
                if len(pair) < 5 or np.allclose(pair["a"], pair["b"]):
                    continue
                stat = wilcoxon(pair["a"], pair["b"])
                tests.append({"metric": c, "A": base, "B": other, "n_configs": len(pair),
                              "median_A": pair["a"].median(), "median_B": pair["b"].median(),
                              "p_value": stat.pvalue})
        tests = pd.DataFrame(tests)
        if len(tests):
            print(tests.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
            tests.to_csv(args.out_dir / "prediction_dynamics_tests.csv", index=False)
    elif wilcoxon is None:
        print("\n[info] scipy が無いため検定を省略しました")
    print(f"\n[OK] {args.out_dir}")


if __name__ == "__main__":
    main()
