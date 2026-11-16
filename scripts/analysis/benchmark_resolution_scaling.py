# scripts/analysis/benchmark_resolution_scaling.py
"""
ボクセル解像度 K ごとに、拡散モデルの推論時間と GPU メモリを測る (査読指摘 E.7 / R3.2)。

計測対象
  1. 解像度の比較 (未学習のネットワーク):
     K ごとに、学習済みモデルと同じ構造の U-Net + GaussianDiffusion を組み立てて測る。
     計算時間とメモリはネットワークの構造と入力サイズで決まり、重みの値には依存しないため、
     学習していない解像度 (中間の K) も測れる。--dims で U-Net の基本幅を指定でき、
     幅を固定して K だけを変える比較ができる。
  2. 学習済みモデル (--trained):
     実際に評価で使ったモデルも同じ条件で測る。未学習のネットワークと同じ値になることを
     確かめれば、1 の前提の裏付けになる。

計測内容 (1 回の計画ステップ = M 枚のサンプル生成、実際の評価と同じ predict を呼ぶ)
  - 推論時間: ウォームアップ後に repeats 回測った平均と標準偏差 (GPU 同期あり)
  - GPU メモリ: torch.cuda.max_memory_allocated (PyTorch が確保したテンソルの最大量) と
                max_memory_reserved (キャッシュを含めて確保した最大量)、モデルの重みの量
  - メモリ不足 (OOM) の場合は status=OOM として記録し、次の設定に進む

解像度と画像サイズの対応
  K 枚の断面 (各 K x K) を sqrt(K) x sqrt(K) に並べた画像を使うので、K は平方数に限る。
  画像の一辺は K * sqrt(K) で、U-Net の制約 (8 の倍数) に合わせて切り上げる
  (実際の評価と同じく、条件画像は推論器の中で 0 埋めされる)。
    K=16 -> 64 (64),  K=25 -> 125 (128),  K=36 -> 216 (216),  K=49 -> 343 (344)

使い方 (リポジトリのルートで):
  python scripts/analysis/benchmark_resolution_scaling.py \
    --resolutions 16,25,36,49 --dims 64,128 \
    --trained 16=<K=16 の学習済みモデル>,49=<K=49 の学習済みモデル> \
    --out_csv analysis/revise/scalability/resolution_scaling_rtx4090.csv
"""
from __future__ import annotations

import argparse
import gc
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from denoising_diffusion_pytorch.policy.inference.conditional_diffusion_slice_image_inferencer import (
    ConditionalDiffusionSliceImageInferencer,
)
from denoising_diffusion_pytorch.policy.types import PlanningPolicyInput

UNET_DOWNSAMPLE = 8   # dim_mults=(1,2,4,8) で 3 回ダウンサンプリングするため
MiB = 1024 ** 2


# ---------------------------------------------------------------------------
# 解像度とモデルの準備
# ---------------------------------------------------------------------------
def image_side(K: int) -> int:
    r = math.isqrt(K)
    if r * r != K:
        raise ValueError(f"K={K}: K は平方数である必要があります (16, 25, 36, 49, ...)")
    return K * r


def model_side(K: int) -> int:
    s = image_side(K)
    return int(math.ceil(s / UNET_DOWNSAMPLE) * UNET_DOWNSAMPLE)


class _EMAWrapper:
    """推論器が要求する ema_model 属性だけを持つ入れ物 (未学習のネットワーク用)。"""
    def __init__(self, model):
        self.ema_model = model


def build_untrained(K: int, dim: int, device: str, sampling_timesteps: int):
    from denoising_diffusion_pytorch.models.unet_2d_simple_devel2 import Unet
    from denoising_diffusion_pytorch.models.conditional_image_diffusion_cfg_devel2 import GaussianDiffusion

    size = model_side(K)
    # 学習済みモデル (unet_D64 / unet_D128) と同じ構造・同じ拡散の設定
    net = Unet(dim=dim, dim_mults=(1, 2, 4, 8), flash_attn=True, self_condition=False, mask_dim=size)
    diffusion = GaussianDiffusion(
        model=net, image_size=size,
        timesteps=1000, sampling_timesteps=sampling_timesteps, beta_schedule="sigmoid",
    )
    diffusion = diffusion.to(device).eval()
    return _EMAWrapper(diffusion)


def load_trained(run_dir: str, epoch: str, device: str):
    from app.wiring.loaders.conditional_diffusion_assets_loader import ConditionalDiffusionAssetsLoader
    from app.wiring.loaders.saved_run_config_loader import SavedRunConfigLoader
    from app.wiring.loaders.checkpoint_path_resolver import CheckpointPathResolver

    loader = ConditionalDiffusionAssetsLoader(
        config_loader=SavedRunConfigLoader(),
        checkpoint_path_resolver=CheckpointPathResolver(),
    )
    assets = loader.load(run_dir=run_dir, epoch=epoch, device=device, infer_model="conditional_diffusion")
    return assets.inferencer


def count_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# ---------------------------------------------------------------------------
# 計測
# ---------------------------------------------------------------------------
def build_dummy_condition(side: int, device: str, observed_patch: int = 4) -> torch.Tensor:
    """既存の benchmark_diffusion_inference_time.py と同じダミー条件 (小さな観測領域あり)。"""
    cond = torch.full((3, side, side), -1.0, dtype=torch.float32, device=device)
    p = min(observed_patch, side)
    cond[:, :p, :p] = 0.0
    return cond


def free_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def measure(wrapper, K: int, M: int, S: int, w: float, device: str, warmup: int, repeats: int) -> dict:
    model = wrapper.ema_model
    infer = ConditionalDiffusionSliceImageInferencer(
        inferencer=wrapper, sample_image_num=M, control_mode="conditional",
        guidance_scale=w, sampling_timesteps=S,
    )
    planning_input = PlanningPolicyInput(normalized_cond=build_dummy_condition(image_side(K), device))

    torch.cuda.synchronize()
    weights_alloc = torch.cuda.memory_allocated()

    for _ in range(warmup):
        infer.predict(planning_input)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        infer.predict(planning_input)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times = np.asarray(times)

    return {
        "status": "ok",
        "mean_time_sec": float(times.mean()),
        "std_time_sec": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / MiB,
        "peak_reserved_MiB": torch.cuda.max_memory_reserved() / MiB,
        "weights_allocated_MiB": weights_alloc / MiB,
        "num_params": count_params(model),
        "all_times_sec": times.tolist(),
    }


def run_one(label: str, make_wrapper, K: int, dim: int | None, args) -> dict:
    row = {
        "model": label, "K": K, "image_side": image_side(K), "model_side": model_side(K),
        "unet_dim": dim, "M": args.M, "S": args.S, "guidance_scale": args.guidance_scale,
        "warmup": args.warmup, "repeats": args.repeats,
    }
    free_cuda()
    wrapper = None
    try:
        wrapper = make_wrapper()
        row.update(measure(wrapper, K, args.M, args.S, args.guidance_scale,
                           args.device, args.warmup, args.repeats))
    except torch.cuda.OutOfMemoryError:
        row.update({"status": "OOM"})
    except Exception as e:  # noqa: BLE001  (学習済みモデルが見つからない等。他の計測は続ける)
        row.update({"status": f"error: {type(e).__name__}: {e}"})
    finally:
        del wrapper
        free_cuda()
    msg = (f"{row['mean_time_sec']:.3f} s, peak {row['peak_allocated_MiB']:.0f} MiB"
           if row["status"] == "ok" else row["status"])
    print(f"[{label:<9}] K={K:>2} side={row['model_side']:>3} dim={dim}: {msg}", flush=True)
    return row


def parse_trained(text: str | None) -> dict[int, str]:
    out = {}
    for item in (text or "").split(","):
        if item.strip():
            k, path = item.split("=", 1)
            out[int(k)] = path.strip()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolutions", type=str, default="16,25,36,49", help="K のリスト (平方数)")
    p.add_argument("--dims", type=str, default="64,128", help="未学習の U-Net の基本幅のリスト")
    p.add_argument("--trained", type=str, default=None,
                   help="学習済みモデル: K=パス をカンマ区切り (例: 16=/path/a,49=/path/b)")
    p.add_argument("--epoch", type=str, default="100000")
    p.add_argument("--M", type=int, default=32, help="計画ステップあたりのサンプル数")
    p.add_argument("--S", type=int, default=20, help="DDIM のステップ数")
    p.add_argument("--guidance_scale", type=float, default=1.2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_csv", type=Path, required=True)
    args = p.parse_args()

    torch.cuda.set_device(args.device)
    env = {
        "gpu": torch.cuda.get_device_name(args.device),
        "gpu_total_MiB": torch.cuda.get_device_properties(args.device).total_memory / MiB,
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
    }
    print(f"[env] {env}")

    rows = []
    for K, path in sorted(parse_trained(args.trained).items()):
        if not Path(path).is_absolute() or not (Path(path) / "config_resolved.yaml").exists():
            print(f"[warn] 学習済みモデルのフォルダが見つかりません (K={K}): {path!r} -> この計測は error として記録します")
        rows.append(run_one("trained", lambda path=path: load_trained(path, args.epoch, args.device),
                            K, None, args))
        rows[-1]["train_run_dir"] = path
    for dim in [int(x) for x in args.dims.split(",") if x.strip()]:
        for K in [int(x) for x in args.resolutions.split(",") if x.strip()]:
            rows.append(run_one("untrained",
                                lambda K=K, dim=dim: build_untrained(K, dim, args.device, args.S),
                                K, dim, args))

    df = pd.DataFrame(rows)
    for k, v in env.items():
        df[k] = v
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    cols = ["model", "unet_dim", "K", "model_side", "status", "mean_time_sec", "std_time_sec",
            "peak_allocated_MiB", "peak_reserved_MiB", "weights_allocated_MiB", "num_params"]
    print("\n" + df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"\n[OK] {args.out_csv}")


if __name__ == "__main__":
    main()
