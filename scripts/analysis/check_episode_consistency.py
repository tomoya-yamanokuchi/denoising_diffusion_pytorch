# scripts/analysis/check_episode_consistency.py
"""
配置 (configuration) を1つ固定したとき、6 エピソードの結果が毎回同じになるかを手法ごとに確認する。

(1) 指標による確認 (--metrics):
    aggregate_by_configuration.py の per_configuration_metrics.csv を読み、配置ごとに
    3 指標 (切断誤差・部品残存率・占有率) の最小値と最大値が一致するかを調べる。
    一致すれば、その配置の全エピソードで最終的な結果が同じだったことになる。
    手法 × 分布ごとに「全エピソード同じ結果だった配置の数」と、エピソード間の標準偏差の平均・最大を表示する。

(2) 行動の列による確認 (--runs、任意):
    評価のログ (各ステップの <k>_cost_map_logs.pickle の slice_range = 実際に選ばれた切断) を読み、
    配置ごとに、全エピソードで選ばれた切断の列が完全に一致するかを調べる。
    (1) は結果が同じであることしか言えないが、(2) は「毎ステップ同じ切断が選ばれた」ことまで確かめられる。

使い方:
  python scripts/analysis/check_episode_consistency.py \
    --metrics "simple=<…>/per_configuration_metrics.csv" "complex=<…>/per_configuration_metrics.csv"

  python scripts/analysis/check_episode_consistency.py \
    --runs "VAEAC-simple-A=<…/simple_eval_v1_A_…_revision/epsilon_greedy_00>" ...
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["cutting_error_volume", "part_remaining_rate", "part_occupancy_rate"]
ORDER = ["Random", "VAEAC", "PCD-DM", "Proposed-Nocond", "Proposed", "Proposed-GT"]
STEP_LOG = re.compile(r"^(\d+)_cost_map_logs\.pickle$")


def check_metrics(label: str, path: Path, tol: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    same = np.ones(len(df), bool)
    for m in METRICS:
        same &= (df[f"{m}_max"] - df[f"{m}_min"]).abs() <= tol
    df["identical"] = same
    rows = []
    for (method, obj), g in df.groupby(["method", "object"]):
        rows.append({
            "setting": label, "method": method, "object": obj,
            "configs": len(g),
            "identical_configs": int(g["identical"].sum()),
            "mean_within_std_occ": g["part_occupancy_rate_std"].mean(),
            "max_within_std_occ": g["part_occupancy_rate_std"].max(),
            "max_within_std_cutvol": g["cutting_error_volume_std"].max(),
        })
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], [m for m in ORDER if m in set(out["method"])] +
                                   sorted(set(out["method"]) - set(ORDER)), ordered=True)
    return out.sort_values(["method", "object"])


def action_sequence(ep: Path) -> tuple:
    steps = sorted((int(STEP_LOG.match(f.name).group(1)), f) for f in ep.glob("*_cost_map_logs.pickle"))
    seq = []
    for _, f in steps:
        with f.open("rb") as fh:
            sr = pickle.load(fh).get("slice_range")
        seq.append(tuple(int(v) for v in (sr if sr is not None else [])))
    return tuple(seq)


def check_runs(label: str, run: Path) -> None:
    cases = sorted(d for d in run.iterdir() if d.is_dir() and any(d.glob("episode_*")))
    n_same = 0
    print(f"\n######## {label}: {run}")
    for case in cases:
        eps = sorted(case.glob("episode_*"), key=lambda d: int(d.name.split("_")[1]))
        seqs = [action_sequence(e) for e in eps]
        if not seqs or not seqs[0]:
            print(f"  {case.name}: 行動の記録 (slice_range) が見つかりません"); continue
        same = all(s == seqs[0] for s in seqs[1:])
        n_same += same
        n_unique = len(set(seqs))
        print(f"  {case.name:32s} episodes={len(seqs)}  行動の列が全エピソードで一致: {same}"
              + ("" if same else f"  (異なる列の数 {n_unique})"))
    print(f"  => 行動の列が全エピソードで一致した配置: {n_same}/{len(cases)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", nargs="*", default=[], help="ラベル=per_configuration_metrics.csv")
    ap.add_argument("--runs", nargs="*", default=[], help="ラベル=手法の結果フォルダ (…/epsilon_greedy_00 など)")
    ap.add_argument("--tol", type=float, default=1e-9, help="最小値と最大値を同じとみなす許容差")
    args = ap.parse_args()

    if args.metrics:
        pd.set_option("display.width", 200)
        res = pd.concat([check_metrics(*s.split("=", 1)[:1], Path(s.split("=", 1)[1]), args.tol)
                         for s in args.metrics], ignore_index=True)
        print("=== 配置を固定したとき、6 エピソードの結果 (3 指標すべて) が同じだった配置の数 ===")
        for setting, g in res.groupby("setting", sort=False):
            print(f"\n[{setting}]")
            print(g.drop(columns="setting").round(3).to_string(index=False))
        print("\n=== 手法ごとのまとめ (全分布) ===")
        tot = res.groupby(["setting", "method"], observed=True)[["configs", "identical_configs"]].sum()
        print(tot.to_string())
    for spec in args.runs:
        label, path = spec.split("=", 1)
        check_runs(label, Path(path))


if __name__ == "__main__":
    main()
