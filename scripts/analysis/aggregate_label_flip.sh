#!/usr/bin/env bash
# scripts/analysis/aggregate_label_flip.sh
#
# 観測した切断面の部品ラベルの誤り (査読指摘 E.6 / R1.4) の評価を、誤りの割合ごとに集計し、
# 表 (label_flip_summary.csv) と図 (label_flip.{pdf,png}) を作る。
#   Proposed      : 簡易形状の代表配置 (Object A-C)、誤りの割合 0, 1, 2, 5, 10 %
#                   (結果フォルダのタグ revision_labelflip0p00 ... revision_labelflip0p10)
#   Proposed-Nocond: 観測を使わないので誤りの影響を受けない。論文の評価の結果を基準として使う
#
# 使い方 (リポジトリのルートで、コンテナの中):
#   bash scripts/analysis/aggregate_label_flip.sh
set -euo pipefail

EVAL=${EVAL:-/home/dev/workspace/dataset/nedo_dismantling_log/eval}
OUT=${OUT:-$EVAL/summary_label_flip_simple_representative}
U=$EVAL/unet_D64_T1000_S20_simple_2d_20260605_133339
PY=${PYTHON:-$(command -v python3 || command -v python)}
mkdir -p "$OUT"

declare -A TAG=( [0.0]=0p00 [0.01]=0p01 [0.02]=0p02 [0.05]=0p05 [0.1]=0p10 )
SPECS=()
for rate in 0.0 0.01 0.02 0.05 0.1; do
  t=${TAG[$rate]}
  root=$(ls -d "$U"/*revision_labelflip${t} 2>/dev/null | head -1 || true)
  if [ -z "$root" ]; then echo "[warn] 割合 $rate の結果フォルダが見つかりません (タグ revision_labelflip${t})"; continue; fi
  echo "=== flip_rate=$rate  $root"
  "$PY" scripts/analysis/aggregate_by_configuration.py --roots "$root" --out-dir "$OUT/rate_$t" \
      --label-map "conditional_diffusion:epsilon_greedy_00=Proposed" > "$OUT/rate_$t.log"
  grep -E "^\[read\]" "$OUT/rate_$t.log" || true
  SPECS+=("$rate=$OUT/rate_$t/per_distribution_summary.csv")
done

NOCOND=$(ls -d "$U"/simple_paper_T8_N6_*_proposed_ABC_Nocond 2>/dev/null | head -1 || true)
if [ -n "$NOCOND" ]; then
  echo "=== Proposed-Nocond (基準)  $NOCOND"
  "$PY" scripts/analysis/aggregate_by_configuration.py --roots "$NOCOND" --out-dir "$OUT/nocond" \
      --label-map "conditional_diffusion:no_cond=Proposed-Nocond" > "$OUT/nocond.log"
  grep -E "^\[read\]" "$OUT/nocond.log" || true
fi

"$PY" - "$OUT" "${SPECS[@]}" <<'EOF'
import sys
from pathlib import Path
import pandas as pd

out = Path(sys.argv[1])
rows = []
for spec in sys.argv[2:]:
    rate, path = spec.split("=", 1)
    d = pd.read_csv(path); d.insert(0, "flip_rate", float(rate)); rows.append(d)
df = pd.concat(rows, ignore_index=True)
nocond = pd.read_csv(out / "nocond" / "per_distribution_summary.csv") if (out / "nocond" / "per_distribution_summary.csv").exists() else None

cols = ["flip_rate", "object", "num_episodes_with_cut_error", "num_episodes_total",
        "cutting_error_volume_mean", "part_remaining_rate_mean", "part_occupancy_rate_mean"]
# 代表配置 1 件なので、標準偏差はエピソード間のもの (std_within_mean) を使う
for m in ["cutting_error_volume", "part_remaining_rate", "part_occupancy_rate"]:
    if f"{m}_std_within_mean" in df.columns:
        cols.append(f"{m}_std_within_mean")
summary = df[cols].sort_values(["object", "flip_rate"])
summary.to_csv(out / "label_flip_summary.csv", index=False)
pd.set_option("display.width", 200)
print("\n=== Proposed (誤りの割合ごと) ===")
show = summary.copy()
show["err_freq"] = show["num_episodes_with_cut_error"].astype(int).astype(str) + "/" + show["num_episodes_total"].astype(int).astype(str)
print(show[["object", "flip_rate", "err_freq", "cutting_error_volume_mean", "part_remaining_rate_mean",
            "part_occupancy_rate_mean"]].round(2).to_string(index=False))
if nocond is not None:
    print("\n=== Proposed-Nocond (誤りの影響を受けない基準) ===")
    print(nocond[["object", "num_episodes_with_cut_error", "num_episodes_total", "cutting_error_volume_mean",
                  "part_remaining_rate_mean", "part_occupancy_rate_mean"]].round(2).to_string(index=False))

print(f"\n[OK] {out}/label_flip_summary.csv")
EOF

# 図 (Proposed-Nocond の占有率と誤切断の頻度を点線で重ねる)
"$PY" scripts/analysis/plot_label_flip.py \
  --summary "$OUT/label_flip_summary.csv" \
  --nocond "$OUT/nocond/per_distribution_summary.csv" \
  --out "$OUT/label_flip.pdf"
