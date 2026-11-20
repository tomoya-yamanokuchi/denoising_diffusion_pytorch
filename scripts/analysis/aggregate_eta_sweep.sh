#!/usr/bin/env bash
# scripts/analysis/aggregate_eta_sweep.sh
#
# 複雑形状の代表配置 (Object D/E/F) で η を変えた評価 (査読指摘 E.3) を、η ごとに集計し、
# 誤切断の頻度と占有率のトレードオフの図を描く。
#
#   η = 0.0, 0.25, 0.75, 1.0 : 新しい評価 (結果フォルダのタグが revision のもの)
#   η = 0.5                  : 論文の Table 4 の評価 (complex_paper_T8_N6_*_complex_DEF)。
#                              Random と Proposed-GT もここに含まれ、図では 1 点になる
#
# 使い方 (リポジトリのルートで、コンテナの中):
#   bash scripts/analysis/aggregate_eta_sweep.sh
# 出力:
#   $OUT/eta<値>/per_distribution_summary.csv など (η ごとの集計)
#   $OUT/tradeoff_representative.pdf / .png     (トレードオフの図)
set -euo pipefail

EVAL=${EVAL:-/home/dev/workspace/dataset/nedo_dismantling_log/eval}
# 対話型シェルの python は別名 (alias) のことがあり、スクリプトの中では使えないため python3 を使う
PY=${PYTHON:-$(command -v python3 || command -v python)}
OUT=${OUT:-$EVAL/summary_eta_sweep_representative}
LABELS="conditional_diffusion:epsilon_greedy_00=Proposed,conditional_diffusion:no_cond=Proposed-Nocond,conditional_diffusion:oracle_obs=Proposed-GT,conditional_diffusion:random=Random,diffusion_1d:epsilon_greedy_00=PCD-DM,vaeac:epsilon_greedy_00=VAEAC"
ORDER="Random,VAEAC,PCD-DM,Proposed-Nocond,Proposed,Proposed-GT"

# 結果フォルダの名前の η の表記 (0.25 -> 0p25)
tag() { echo "$1" | sed 's/\./p/'; }

mkdir -p "$OUT"
SUMMARIES=()
for eta in 0.0 0.25 0.5 0.75 1.0; do
  t=$(tag "$eta")
  if [ "$eta" = "0.5" ]; then
    mapfile -t ROOTS < <(ls -d "$EVAL"/*/complex_paper_T8_N6_*_complex_DEF 2>/dev/null)
    expected=6   # Proposed, Nocond, GT, Random, VAEAC, PCD-DM
  else
    mapfile -t ROOTS < <(ls -d "$EVAL"/*/complex_paper_T8_N6_*eta${t}_*_revision 2>/dev/null)
    expected=3   # U-Net (Proposed と Nocond が同じフォルダ), VAEAC, PCD-DM
  fi
  echo "=== eta=$eta  結果フォルダ ${#ROOTS[@]} 個 (想定 $expected 個)"
  printf '    %s\n' "${ROOTS[@]}"
  if [ "${#ROOTS[@]}" -ne "$expected" ]; then
    echo "[warn] eta=$eta の結果フォルダの数が想定と違います。PRO6000 の結果のコピー漏れや、余分なランがないか確認してください"
  fi
  "$PY" scripts/analysis/aggregate_by_configuration.py \
    --roots "${ROOTS[@]}" --out-dir "$OUT/eta$t" \
    --label-map "$LABELS" --method-order "$ORDER" > "$OUT/eta$t.log"
  grep -E "^\[read\]|warn|未完了" "$OUT/eta$t.log" || true
  SUMMARIES+=("$eta=$OUT/eta$t/per_distribution_summary.csv")
done

for ext in pdf png; do
  "$PY" scripts/analysis/plot_tradeoff.py \
    --summary "${SUMMARIES[@]}" \
    --objects "Object D,Object E,Object F" \
    --out "$OUT/tradeoff_representative.$ext"
done

# η ごとの集計をまとめた表 (確認用)
"$PY" - "$OUT" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
out = Path(sys.argv[1])
rows = []
for d in sorted(out.glob("eta*/per_distribution_summary.csv")):
    df = pd.read_csv(d)
    df.insert(0, "eta", d.parent.name.replace("eta", "").replace("p", "."))
    rows.append(df)
allv = pd.concat(rows, ignore_index=True)
allv["err_freq"] = allv["num_episodes_with_cut_error"].astype(int).astype(str) + "/" + allv["num_episodes_total"].astype(int).astype(str)
cols = ["eta", "object", "method", "err_freq", "cutting_error_volume_mean", "part_remaining_rate_mean", "part_occupancy_rate_mean"]
allv[cols].to_csv(out / "eta_sweep_summary.csv", index=False)
pd.set_option("display.width", 200)
for obj, g in allv.groupby("object"):
    print(f"\n[{obj}]")
    print(g[cols[2:] + ["eta"]].sort_values(["method", "eta"]).round(2).to_string(index=False))
print(f"\n[OK] {out}/eta_sweep_summary.csv")
EOF
echo "[OK] $OUT/tradeoff_representative.{pdf,png}"
