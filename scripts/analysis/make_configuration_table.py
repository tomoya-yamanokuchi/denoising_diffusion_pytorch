# scripts/analysis/make_configuration_table.py
"""
aggregate_by_configuration.py の出力 (per_distribution_summary.csv) から、
論文用の手法比較の表 (LaTeX) を作る。

表の形
  - Object ごとの小表 (a), (b), ... を縦に並べる
  - 列: Cut Err. Vol. / Cut Err. Freq. / Part Remain. Rate / Part Occ. Rate
        (Cut Err. Freq. は誤切断があったエピソード数 / 総エピソード数。それ以外は配置間の 平均 ± 標準偏差)
  - 参考値の手法 (既定: Proposed-GT) は最下段に置き、線で区切る
  - 太字 (--bold-rule で選ぶ。どちらも参考値の手法は除く):
      occupancy (既定): 誤切断の頻度の列は、最も少ない値 (同点はすべて) を太字にする。
                        占有率の列は、誤切断の頻度が最も少ない手法の中で、最も高い値を太字に
                        する。誤切断がゼロの手法がある場合、占有率の太字は従来の論文の
                        ルール (切断誤差がない手法の中で占有率が最も高い値) と同じになる
      safety          : 安全性の指標 (切断誤差・部品残存率・誤切断の頻度) で最も良い値
      none            : 太字なし (安全性と占有率のトレードオフは本文や図で比較する場合)

使い方 (リポジトリのルートで):
  python scripts/analysis/make_configuration_table.py \
    --summary <集計の出力先>/per_distribution_summary.csv \
    --method-order "Random,VAEAC,PCD-DM,Proposed-Nocond,Proposed,Proposed-GT" \
    --out <出力先>/table_complex_configurations.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = [
    # (列名, CSV の平均の列, CSV の標準偏差の列, 良い向き)
    ("Cut Err. Vol. $\\downarrow$", "cutting_error_volume_mean", "cutting_error_volume_std_between", "min"),
    ("Part Remain. Rate $\\uparrow$", "part_remaining_rate_mean", "part_remaining_rate_std_between", "max"),
    ("Part Occ. Rate $\\uparrow$", "part_occupancy_rate_mean", "part_occupancy_rate_std_between", "max"),
]
FREQ_HEADER = "Cut Err. Freq. $\\downarrow$"
TOL = 1e-9


def best_mask(values: pd.Series, direction: str) -> pd.Series:
    target = values.min() if direction == "min" else values.max()
    return (values - target).abs() <= TOL


def bold(text: str, on: bool) -> str:
    return f"\\textbf{{{text}}}" if on else text


def fmt_pm(mean: float, std: float, digits: int, on: bool) -> str:
    m, s = f"{mean:.{digits}f}", f"{std:.{digits}f}"
    return f"$\\mathbf{{{m}}} \\pm \\mathbf{{{s}}}$" if on else f"${m} \\pm {s}$"


def object_block(df: pd.DataFrame, obj: str, letter: str, methods: list[str], reference: set[str],
                 digits: int, bold_rule: str, display: str | None = None) -> list[str]:
    display = display or obj
    g = df[df["object"] == obj].set_index("method").reindex(methods).dropna(how="all")
    comp = g[~g.index.isin(reference)]
    freq_rate = comp["num_episodes_with_cut_error"] / comp["num_episodes_total"]
    none = pd.Series(False, index=g.index)
    marks = {m: none for _, m, _, _ in METRICS}
    marks["freq"] = none
    if bold_rule == "none":
        pass
    elif bold_rule == "safety":
        for _, mean_col, _, direction in METRICS:
            if mean_col != "part_occupancy_rate_mean":
                marks[mean_col] = best_mask(comp[mean_col], direction).reindex(g.index, fill_value=False)
        marks["freq"] = best_mask(freq_rate, "min").reindex(g.index, fill_value=False)
    else:  # occupancy: 誤切断の頻度の最良値 + その手法の中で占有率が最も高い値
        marks["freq"] = best_mask(freq_rate, "min").reindex(g.index, fill_value=False)
        safest = comp[best_mask(freq_rate, "min")]
        occ = best_mask(safest["part_occupancy_rate_mean"], "max")
        marks["part_occupancy_rate_mean"] = occ.reindex(g.index, fill_value=False)

    lines = [f"\\multicolumn{{5}}{{c}}{{({letter}) {display}}} \\\\", "\\midrule",
             "Method & " + " & ".join([METRICS[0][0], FREQ_HEADER] + [h for h, *_ in METRICS[1:]]) + " \\\\",
             "\\midrule"]
    ref_started = False
    for method, r in g.iterrows():
        if method in reference and not ref_started:
            lines.append("\\midrule")
            ref_started = True
        vals = [fmt_pm(r[mean_col], r[std_col], digits, bool(marks[mean_col][method]))
                for _, mean_col, std_col, _ in METRICS]
        freq = bold(f"{int(r['num_episodes_with_cut_error'])}/{int(r['num_episodes_total'])}",
                    bool(marks["freq"][method]))
        # 列の順: Cut Err. Vol. | Cut Err. Freq. | Part Remain. Rate | Part Occ. Rate
        cells = [method, vals[0], freq] + vals[1:]
        lines.append(" & ".join(cells) + " \\\\")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", type=Path, required=True, help="per_distribution_summary.csv")
    p.add_argument("--method-order", type=str, required=True, help="表の手法の並び順 (カンマ区切り)")
    p.add_argument("--reference", type=str, default="Proposed-GT",
                   help="参考値として最下段に区切って置く手法 (カンマ区切り)。太字の比較から除く")
    p.add_argument("--objects", type=str, default=None, help="表に入れる Object の順 (省略時は CSV の順)")
    p.add_argument("--display-names", type=str, default=None,
                   help="小表の見出しの表示名: 'Object A=Distribution A,...' (CSV の名前は変えない)")
    p.add_argument("--digits", type=int, default=2)
    p.add_argument("--bold-rule", choices=["occupancy", "safety", "none"], default="occupancy",
                   help="occupancy: 誤切断の頻度が最も少ない手法の中で占有率が最も高い値 (既定) / "
                        "safety: 安全性の指標で最も良い値 / none: 太字なし")
    p.add_argument("--caption", type=str, default="Task performance comparison of the complex-shaped models.")
    p.add_argument("--label", type=str, default="tab:complex_baseline_comparison")
    p.add_argument("--out", type=Path, required=True, help="出力する .tex ファイル")
    args = p.parse_args()

    df = pd.read_csv(args.summary)
    required = {"method", "object", "num_episodes_with_cut_error", "num_episodes_total"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV に必要な列がありません: {sorted(missing)} "
                         "(aggregate_by_configuration.py の最新版で集計し直してください)")

    methods = [m.strip() for m in args.method_order.split(",") if m.strip()]
    reference = {m.strip() for m in args.reference.split(",") if m.strip()}
    objects = ([o.strip() for o in args.objects.split(",")] if args.objects
               else list(dict.fromkeys(df["object"])))
    unknown = sorted(set(df["method"]) - set(methods))
    if unknown:
        print(f"[warn] --method-order に無い手法は表に入りません: {unknown}")

    display_names = dict(item.split("=", 1) for item in (args.display_names or "").split(",") if "=" in item)
    body = []
    for i, obj in enumerate(objects):
        if i:
            body += ["\\midrule", "\\addlinespace"]
        body += object_block(df, obj, chr(ord("a") + i), methods, reference, args.digits, args.bold_rule,
                             display_names.get(obj))

    tex = "\n".join([
        "% 自動生成: scripts/analysis/make_configuration_table.py",
        f"% 入力: {args.summary}",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{args.caption}}}",
        f"\\label{{{args.label}}}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()
