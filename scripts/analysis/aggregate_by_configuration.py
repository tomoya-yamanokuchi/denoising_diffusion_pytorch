# scripts/analysis/aggregate_by_configuration.py
"""
配置分布 (Object) ごとに、複数の内部構造配置 (configuration) にまたがる
平均とばらつきを集計する。査読指摘「各配置分布につき 10〜20 個の配置を評価し、
配置間の平均とばらつきを報告せよ」への対応用。

集計は 2 段階:
  1. 配置ごと   : 各ケース (例 objD_type1_sample_003) の N エピソードを平均する
  2. 分布ごと   : Object ごとに、配置ごとの平均値の 平均 と 標準偏差 (配置間) をとる
                   あわせて、配置内 (エピソード間) の標準偏差の平均も出す

エピソードの読み込みは aggregate_task_metrics.py の aggregate_single_root を
そのまま使う (指標の定義・標準偏差の ddof は既存の集計と同じ)。

使い方 (リポジトリのルートで):
  python scripts/analysis/aggregate_by_configuration.py \
      --roots <評価ログ>/complex_eval_v1_*_revision \
      --out-dir <出力先>

  * --roots には、condition_metadata.yaml があるフォルダ (…/epsilon_greedy_00)
    か、その親 (ランのフォルダ) を何個でも渡せる。下の階層から自動で探す。
  * 手法ごとに名前を付けたい場合は  Proposed=<フォルダ>  の形で渡す。
    名前を付けない場合は、メタデータの infer_model と control_mode から付ける。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aggregate_task_metrics import (
    PAPER_METRIC_COLUMNS,
    STD_DDOF,
    aggregate_single_root,
)

CASE_NAME_PATTERN = re.compile(r"^obj([A-Z])_")


# ---------------------------------------------------------------------------
# 入力の解決
# ---------------------------------------------------------------------------
def parse_root_spec(spec: str) -> tuple[str | None, Path]:
    """'Label=/path' または '/path' を (label, path) に分ける。"""
    if "=" in spec and not Path(spec).exists():
        label, path = spec.split("=", 1)
        return label.strip() or None, Path(path)
    return None, Path(spec)


def find_rollout_roots(path: Path) -> list[Path]:
    """condition_metadata.yaml があるフォルダを、path 自身とその下から探す。"""
    if (path / "condition_metadata.yaml").exists():
        return [path]
    found = sorted(p.parent for p in path.rglob("condition_metadata.yaml"))
    if not found:
        raise FileNotFoundError(f"condition_metadata.yaml が見つかりません: {path}")
    return found


def _find_case_list(node: Any) -> list[dict] | None:
    """config_resolved.yaml の中から、name を持つ要素のリスト (評価ケース一覧) を探す。"""
    if isinstance(node, dict):
        cases = node.get("cases")
        if isinstance(cases, dict) and isinstance(cases.get("cases"), list):
            return cases["cases"]
        for v in node.values():
            found = _find_case_list(v)
            if found is not None:
                return found
    return None


def load_case_labels(root: Path) -> dict[str, dict[str, Any]]:
    """ケース名 → {paper_label, sample_id, distribution_id} を config_resolved.yaml から読む。"""
    cfg_path = root / "config_resolved.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cases = _find_case_list(cfg) or []
    out = {}
    for c in cases:
        if isinstance(c, dict) and "name" in c:
            out[str(c["name"])] = {
                "paper_label": c.get("paper_label"),
                "sample_id": c.get("sample_id"),
                "distribution_id": c.get("distribution_id"),
            }
    return out


def object_label_for(case: str, labels: dict[str, dict[str, Any]]) -> str:
    info = labels.get(case) or {}
    if info.get("paper_label"):
        return str(info["paper_label"])
    m = CASE_NAME_PATTERN.match(case)
    if m:
        return f"Object {m.group(1)}"
    return case


def expected_num_episodes(root: Path) -> int | None:
    meta_path = root / "condition_metadata.yaml"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    n = (meta.get("eval") or {}).get("num_episodes")
    return int(n) if n is not None else None


def load_per_episode(root_specs: list[str]) -> pd.DataFrame:
    frames = []
    for spec in root_specs:
        label, path = parse_root_spec(spec)
        for root in find_rollout_roots(path):
            df = aggregate_single_root(root=root)
            labels = load_case_labels(root)
            if label is not None:
                method = label
            elif {"infer_model", "control_mode"} <= set(df.columns):
                method = f"{df['infer_model'].iloc[0]}:{df['control_mode'].iloc[0]}"
            else:
                method = str(df["condition"].iloc[0])
            df.insert(0, "method", method)
            df.insert(1, "object", [object_label_for(c, labels) for c in df["case"]])
            df.insert(2, "sample_id", [(labels.get(c) or {}).get("sample_id") for c in df["case"]])
            df["rollout_root"] = str(root)
            df["expected_num_episodes"] = expected_num_episodes(root)
            frames.append(df)
            print(f"[read] {method:<34} {root}  ({len(df)} episodes, {df['case'].nunique()} cases)")
    if not frames:
        raise RuntimeError("読み込めたエピソードがありません")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------
def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=STD_DDOF)) if len(values) > STD_DDOF else float("nan")


def build_per_configuration(per_episode: pd.DataFrame) -> pd.DataFrame:
    """段階1: 配置 (ケース) ごとに、エピソードの平均・標準偏差・最小・最大。"""
    rows = []
    for (method, obj, case), g in per_episode.groupby(["method", "object", "case"], sort=True):
        exp = g["expected_num_episodes"].iloc[0]
        row = {
            "method": method,
            "object": obj,
            "case": case,
            "sample_id": g["sample_id"].iloc[0],
            "num_episodes": int(len(g)),
            "expected_num_episodes": None if pd.isna(exp) else int(exp),
        }
        row["complete"] = bool(row["expected_num_episodes"] is None
                               or row["num_episodes"] == row["expected_num_episodes"])
        for m in PAPER_METRIC_COLUMNS:
            v = g[m].to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.mean(v))
            row[f"{m}_std"] = _std(v)
            row[f"{m}_min"] = float(np.min(v))
            row[f"{m}_max"] = float(np.max(v))
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_distribution(per_config: pd.DataFrame) -> pd.DataFrame:
    """段階2: Object ごとに、配置ごとの平均値の 平均 と 配置間の標準偏差。"""
    rows = []
    for (method, obj), g in per_config.groupby(["method", "object"], sort=True):
        row = {
            "method": method,
            "object": obj,
            "num_configurations": int(len(g)),
            "num_incomplete_configurations": int((~g["complete"]).sum()),
            "num_episodes_total": int(g["num_episodes"].sum()),
        }
        for m in PAPER_METRIC_COLUMNS:
            cm = g[f"{m}_mean"].to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.mean(cm))                       # 配置の平均値の平均
            row[f"{m}_std_between"] = _std(cm)                          # 配置間の標準偏差
            row[f"{m}_std_within_mean"] = float(np.nanmean(g[f"{m}_std"].to_numpy(dtype=float)))
            row[f"{m}_config_min"] = float(np.min(cm))
            row[f"{m}_config_max"] = float(np.max(cm))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 表示
# ---------------------------------------------------------------------------
METRIC_SHORT = {
    "cutting_error_volume": "CutErrVol",
    "part_remaining_rate": "PartRemain",
    "part_occupancy_rate": "PartOccup",
}


def print_report(per_config: pd.DataFrame, per_dist: pd.DataFrame, digits: int) -> None:
    fmt = f"{{:.{digits}f}}"
    print("\n=== 分布 (Object) ごと: 配置間の 平均 ± 標準偏差  [配置内の標準偏差の平均] ===")
    for _, r in per_dist.iterrows():
        status = "" if r["num_incomplete_configurations"] == 0 else \
            f"  ※未完了の配置 {r['num_incomplete_configurations']} 件を含む"
        print(f"\n{r['method']}  /  {r['object']}   (配置 {r['num_configurations']} 件, "
              f"エピソード計 {r['num_episodes_total']}){status}")
        for m in PAPER_METRIC_COLUMNS:
            print(f"  {METRIC_SHORT.get(m, m):<11} "
                  f"{fmt.format(r[f'{m}_mean'])} ± {fmt.format(r[f'{m}_std_between'])}"
                  f"   [within {fmt.format(r[f'{m}_std_within_mean'])}]"
                  f"   (配置の範囲 {fmt.format(r[f'{m}_config_min'])} 〜 {fmt.format(r[f'{m}_config_max'])})")

    incomplete = per_config[~per_config["complete"]]
    if len(incomplete):
        print("\n[warn] エピソード数が設定に満たない配置 (評価途中の可能性):")
        for _, r in incomplete.iterrows():
            print(f"  {r['method']} / {r['case']}: {r['num_episodes']} / {r['expected_num_episodes']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", required=True,
                   help="評価結果のフォルダ (ランのフォルダか epsilon_greedy_00)。Label=path で手法名を付けられる")
    p.add_argument("--out-dir", type=Path, required=True, help="CSV の出力先")
    p.add_argument("--digits", type=int, default=4, help="画面表示の小数点以下の桁数")
    p.add_argument("--complete-only", action="store_true",
                   help="エピソード数が設定どおりの配置だけを集計に使う")
    args = p.parse_args()

    per_episode = load_per_episode(args.roots)
    per_config = build_per_configuration(per_episode)
    used = per_config[per_config["complete"]] if args.complete_only else per_config
    per_dist = build_per_distribution(used)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_episode.to_csv(args.out_dir / "per_episode_metrics.csv", index=False)
    per_config.to_csv(args.out_dir / "per_configuration_metrics.csv", index=False)
    per_dist.to_csv(args.out_dir / "per_distribution_summary.csv", index=False)

    print_report(per_config, per_dist, args.digits)
    print(f"\n[OK] {args.out_dir}/per_episode_metrics.csv")
    print(f"[OK] {args.out_dir}/per_configuration_metrics.csv")
    print(f"[OK] {args.out_dir}/per_distribution_summary.csv")


if __name__ == "__main__":
    main()
