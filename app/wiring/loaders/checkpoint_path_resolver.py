from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


"""
CheckpointPathResolver は、評価時に「どの checkpoint ファイルを読むか」を決めるための
小さな責務のオブジェクト。

主な意図:
- checkpoint の命名規約や保存規約を PolicyModelAssetsLoader から分離する
- "latest" や epoch 指定を、具体的な checkpoint パスへ解決する
- checkpoint が存在しない場合に、意味のあるエラーを早い段階で出す

なぜ分離するのか:
- loader 本体は、本来
    - saved run config を読む
    - model / method / trainer を再構築する
    - checkpoint をロードして PolicyModelAssets を返す
  ことに集中したい
- その一方で、checkpoint のファイル名規約
    - model-100.pt
    - model-400.pt
    - latest.pt
    - checkpoints/epoch_400.ckpt
  などは、将来変更されうる独立した関心事
- そのため、checkpoint パスの解決を専用オブジェクトに閉じ込めておくと、
  規約変更の影響を局所化できる

役割:
- epoch="latest" のような指定を、実際のファイルパスへ変換する
- epoch=100 のような指定を、対応する checkpoint パスへ変換する
- 対応する checkpoint が存在しない場合は FileNotFoundError を送出する

補足:
- 既存の Trainer.load(epoch) が checkpoint 規約をすでに内部で知っている場合、
  この Resolver は必須ではない
- ただし、命名規約を loader の外へ分離したい場合や、
  明示的に checkpoint パスを扱いたい場合には有用
"""

@dataclass
class CheckpointPathResolver:
    def resolve(self, run_dir: str | Path, epoch: str | int = "latest") -> Path:
        run_dir = Path(run_dir)

        if epoch == "latest":
            candidates = sorted(run_dir.glob("model-*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"No checkpoint matching 'model-*.pt' under: {run_dir}"
                )
            return candidates[-1]

        ckpt_path = run_dir / f"model-{epoch}.pt"


        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        return ckpt_path



