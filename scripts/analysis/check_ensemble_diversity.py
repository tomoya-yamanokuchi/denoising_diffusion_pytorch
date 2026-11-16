# scripts/analysis/check_ensemble_diversity.py
"""
評価ログから、予測サンプルの多様性と、それが行動にどう影響したかを確認する。
VAEAC の結果がエピソード間で標準偏差ゼロになる理由の裏付け用。

各ケース (配置) について、次の3点をステップごとに集計する:
  1. 予測画像の多様性  : raw_pred_image/step_k/ensemble_z_*.png (M 枚) のうち、
                          互いに異なる画像の枚数と、画素ごとの標準偏差の平均
  2. 判定の一致        : k_cost_map_logs.pickle の presence_probability が 0/1 だけか
                          (M 個のサンプルの「目標部品がある/ない」の判定が完全に一致しているか)
  3. 行動の一致        : 同じケースの全エピソードで、各ステップの slice_range が同じか

使い方 (リポジトリのルートで):
  python scripts/analysis/check_ensemble_diversity.py <評価結果のフォルダ> [<フォルダ> ...]

  フォルダには、condition_metadata.yaml があるフォルダ (…/epsilon_greedy_00) か、
  その親 (ランのフォルダ) を渡す。下の階層の episode_* を自動で探す。
  比較のため、Proposed の結果のフォルダも一緒に渡すとよい。
"""
from __future__ import annotations

import hashlib
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

STEP_LOG = re.compile(r"^(\d+)_cost_map_logs\.pickle$")


def load_images(step_dir: Path) -> list[np.ndarray]:
    files = sorted(step_dir.glob("ensemble_z_*.png"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return [np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0 for f in files]


def image_diversity(images: list[np.ndarray]) -> tuple[int, float]:
    """(互いに異なる画像の枚数, 画素ごとの標準偏差の平均) を返す。"""
    if not images:
        return 0, float("nan")
    hashes = {hashlib.md5(im.tobytes()).hexdigest() for im in images}
    stack = np.stack(images)                       # [M, H, W, 3]
    return len(hashes), float(stack.std(axis=0).mean())


def presence_is_binary(log: dict) -> bool | None:
    """presence_probability が全軸で 0/1 だけなら True。"""
    try:
        pp = log["brier_score"]["presence_probability"]
    except (KeyError, TypeError):
        return None
    vals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in pp.values()])
    return bool(np.all(np.isclose(vals, 0.0) | np.isclose(vals, 1.0)))


def find_episode_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("*_cost_map_logs.pickle")
                   if p.parent.name.startswith("episode_")})


def analyze(root: Path) -> None:
    episodes = find_episode_dirs(root)
    if not episodes:
        print(f"[skip] cost_map_logs が見つかりません: {root}")
        return
    by_case: dict[str, list[Path]] = defaultdict(list)
    for ep in episodes:
        by_case[f"{ep.parent.parent.name}/{ep.parent.name}"].append(ep)

    print(f"\n######## {root}")
    total = {"steps": 0, "distinct_gt1": 0, "nonbinary": 0, "action_diff": 0, "action_steps": 0}
    for case, eps in sorted(by_case.items()):
        n_unique, pix_std, binary = [], [], []
        actions: dict[int, list] = defaultdict(list)
        for ep in eps:
            for f in sorted(ep.glob("*_cost_map_logs.pickle")):
                k = int(STEP_LOG.match(f.name).group(1))
                with f.open("rb") as fh:
                    log = pickle.load(fh)
                actions[k].append(log.get("slice_range"))
                b = presence_is_binary(log)
                if b is not None:
                    binary.append(b)
                u, s = image_diversity(load_images(ep / "raw_pred_image" / f"step_{k}"))
                if u:
                    n_unique.append(u)
                    pix_std.append(s)
        same_action = [all(a == acts[0] for a in acts) for _, acts in sorted(actions.items())]

        total["steps"] += len(n_unique)
        total["distinct_gt1"] += sum(u > 1 for u in n_unique)
        total["nonbinary"] += sum(not b for b in binary)
        total["action_steps"] += len(same_action)
        total["action_diff"] += sum(not s for s in same_action)

        print(f"  {case}  ({len(eps)} episodes)")
        if n_unique:
            print(f"    予測画像: 互いに異なる枚数 平均 {np.mean(n_unique):.1f} (最小 {min(n_unique)}, 最大 {max(n_unique)})"
                  f" / 画素ごとの標準偏差 平均 {np.mean(pix_std):.4f}")
        if binary:
            print(f"    判定    : presence_probability が 0/1 だけのステップ {sum(binary)}/{len(binary)}")
        print(f"    行動    : 全エピソードで同じ切断が選ばれたステップ {sum(same_action)}/{len(same_action)}")

    print(f"  -- 合計: 予測画像が2種類以上あったステップ {total['distinct_gt1']}/{total['steps']}, "
          f"判定が一致しなかったステップ {total['nonbinary']}, "
          f"エピソード間で行動が違ったステップ {total['action_diff']}/{total['action_steps']}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        analyze(Path(arg))


if __name__ == "__main__":
    main()
