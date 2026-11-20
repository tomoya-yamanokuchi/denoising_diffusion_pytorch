# scripts/analysis/analyze_risk_decisions.py
"""
評価ログ (k_cost_map_logs.pickle) から、計画ステップごとに
「目標部品の存在の予測」と「切断の判定」を手法間で比べる。

各ステップで集計する内容 (断面 = x / y / z 各軸の候補の切断面):
  p_target      : 正解で目標部品がある断面での、目標部品ありと予測したサンプルの割合 p の平均 [最小, 最大]
  p_nontarget   : 正解で目標部品がない断面での p の平均
  false_safe    : 正解で目標部品があるのに「安全」と判定された断面の数
  cut_slices    : 実際に選ばれた切断で取り除かれた断面の数
  cut_on_target : そのうち、正解で目標部品がある断面の数と、それらの p

ログの項目:
  brier_score.presence_probability / ground_truth_presence (軸ごとの配列)
  false_safe_rate.feasible_mask (軸ごとの配列)
  slice_range (選ばれた切断の大域の番号。g < K -> z, g < 2K -> x, それ以外 -> y、断面番号は g mod K)

使い方 (リポジトリのルートで):
  python scripts/analysis/analyze_risk_decisions.py \
    --episodes "Proposed=<…/epsilon_greedy_00/Object_D>" "Proposed-Nocond=<…/no_cond/Object_D>" \
    [--episode 0]          # 指定しなければ全エピソードを表示
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np

AXES = ("x", "y", "z")
STEP_LOG = re.compile(r"^(\d+)_cost_map_logs\.pickle$")


def axis_of_global(g: int, k: int) -> tuple[str, int]:
    code = g // k
    return {0: "z", 1: "x", 2: "y"}[code], g % k


def step_row(log: dict) -> dict | None:
    bs = log.get("brier_score") or {}
    if not bs.get("available", False):
        return None
    p = {a: np.asarray(bs["presence_probability"][a], float) for a in AXES}
    gt = {a: np.asarray(bs["ground_truth_presence"][a], float) > 0.5 for a in AXES}
    fs = log.get("false_safe_rate") or {}
    feas = {a: np.asarray(fs["feasible_mask"][a], bool) for a in AXES} if "feasible_mask" in fs else None
    k = len(p["x"])

    p_t = np.concatenate([p[a][gt[a]] for a in AXES])
    p_n = np.concatenate([p[a][~gt[a]] for a in AXES])
    row = {
        "n_target_surfaces": int(len(p_t)),
        "p_target_mean": float(p_t.mean()) if len(p_t) else np.nan,
        "p_target_min": float(p_t.min()) if len(p_t) else np.nan,
        "p_target_max": float(p_t.max()) if len(p_t) else np.nan,
        "p_nontarget_mean": float(p_n.mean()) if len(p_n) else np.nan,
        "false_safe": int(sum((feas[a] & gt[a]).sum() for a in AXES)) if feas else -1,
        "feasible": int(sum(feas[a].sum() for a in AXES)) if feas else -1,
    }
    sr = log.get("slice_range")
    idx = [int(v) for v in (sr if sr is not None else [])]
    on_t, p_on = 0, []
    for g in idx:
        a, i = axis_of_global(g, k)
        if gt[a][i]:
            on_t += 1
            p_on.append(p[a][i])
    row.update(cut_slices=len(idx), cut_on_target=on_t,
               p_cut_on_target=(float(np.mean(p_on)) if p_on else np.nan))
    return row


def episode_rows(ep: Path) -> list[tuple[int, dict]]:
    out = []
    for f in sorted(ep.glob("*_cost_map_logs.pickle"), key=lambda f: int(STEP_LOG.match(f.name).group(1))):
        with f.open("rb") as fh:
            r = step_row(pickle.load(fh))
        if r is not None:
            out.append((int(STEP_LOG.match(f.name).group(1)), r))
    return out


def fmt(r: dict) -> str:
    return (f"p_target {r['p_target_mean']:.2f} [{r['p_target_min']:.2f},{r['p_target_max']:.2f}] "
            f"(n={r['n_target_surfaces']:3d})  p_nontarget {r['p_nontarget_mean']:.2f}  "
            f"false_safe {r['false_safe']:3d}/{r['feasible']:3d}  "
            f"cut {r['cut_slices']:2d} slices, on target {r['cut_on_target']:2d}"
            + (f" (p={r['p_cut_on_target']:.2f})" if r['cut_on_target'] else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", nargs="+", required=True,
                    help="手法名=ケースのフォルダ (…/Object_D など。中の episode_* を読む)")
    ap.add_argument("--episode", type=int, default=None, help="表示するエピソード番号 (省略時は全エピソード)")
    args = ap.parse_args()

    for spec in args.episodes:
        label, path = spec.split("=", 1)
        case = Path(path)
        eps = sorted(case.glob("episode_*"), key=lambda d: int(d.name.split("_")[1]))
        if args.episode is not None:
            eps = [e for e in eps if e.name == f"episode_{args.episode}"]
        print(f"\n######## {label}: {case}")
        for ep in eps:
            rows = episode_rows(ep)
            if not rows:
                print(f"  {ep.name}: brier_score の記録がありません"); continue
            print(f"  --- {ep.name}")
            for k, r in rows:
                print(f"    step {k}: {fmt(r)}")


if __name__ == "__main__":
    main()
