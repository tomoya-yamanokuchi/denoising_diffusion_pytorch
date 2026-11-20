"""
観測した切断面の部品ラベルの誤り (知覚の誤りの模擬) を入れる処理。

査読指摘 E.6 / R1.4: 将来の知覚モジュールの誤りに対する感度を調べるため、
観測した切断面のボクセルの部品ラベル (色) を、確率 flip_rate で別の部品のラベルに入れ替える。

- 誤りは、拡散モデルに渡す観測画像 (planning 用の condition image) にだけ入れる。
  評価の正解 (切断誤差などの計算に使う本当の内部構造) は変えない。
- 一度誤ったボクセルは、そのエピソードの間は同じ誤りのまま (観測した切断面の誤りは変わらない)。
  そのため、乱数の場 (どのボクセルを誤らせるか、どのラベルに入れ替えるか) を
  (ケース名, エピソード番号) ごとに1回だけ作り、エピソード内の全ステップで使い回す。
- 入れ替え先は、その物体に現れる部品の色 (正解の内部構造の画像から求める) のうち、
  元のラベル以外から一様に選ぶ。
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ObservationLabelNoiseConfig:
    flip_rate: float = 0.0  # 観測したボクセルのうち、ラベルを入れ替える割合
    seed     : int   = 0

    @property
    def enabled(self) -> bool:
        return self.flip_rate > 0.0


def build_observation_label_noise_config(cfg: Any) -> ObservationLabelNoiseConfig:
    """Hydra の設定 (None でもよい) から設定を作る。"""
    if cfg is None:
        return ObservationLabelNoiseConfig()
    get = (lambda k, d: cfg.get(k, d)) if hasattr(cfg, "get") else (lambda k, d: getattr(cfg, k, d))
    rate = float(get("flip_rate", 0.0))
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"observation_noise.flip_rate must be in [0, 1], got {rate}")
    return ObservationLabelNoiseConfig(flip_rate=rate, seed=int(get("seed", 0)))


def _palette_from_reference(reference_img: np.ndarray, min_count: int = 1) -> np.ndarray:
    """正解の内部構造の画像に現れる部品の色 (黒 = 未観測・空間 を除く) を返す。[n, 3]"""
    flat = np.asarray(reference_img).reshape(-1, 3)
    flat = flat[np.any(flat != 0, axis=1)]
    if len(flat) == 0:
        return np.zeros((0, 3), dtype=np.asarray(reference_img).dtype)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    return colors[counts >= min_count]


@dataclass
class ObservationLabelFlipper:
    config: ObservationLabelNoiseConfig
    _fields: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)

    def _random_fields(self, key: str, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """(ケース, エピソード) ごとに固定した乱数の場: どのボクセルを誤らせるか / 入れ替え先の選び方。"""
        k = f"{key}|{shape}"
        if k not in self._fields:
            rng = np.random.default_rng([self.config.seed, zlib.crc32(key.encode())])
            u = rng.random(shape)                         # u < flip_rate のボクセルを誤らせる
            r = rng.integers(0, np.iinfo(np.int32).max, size=shape)  # 入れ替え先を選ぶ乱数
            self._fields[k] = (u, r)
        return self._fields[k]

    def apply(self, slice_img: np.ndarray, reference_img: np.ndarray, key: str
              ) -> Tuple[np.ndarray, Optional[Dict[str, int]]]:
        """
        slice_img     : 観測画像 (H, W, 3)。黒 (0, 0, 0) は未観測
        reference_img : 正解の内部構造の画像 (H, W, 3)。部品の色の候補を求めるためだけに使う
        key           : 乱数の場を固定する単位 (例: "<case>/episode_<k>")
        戻り値: (誤りを入れた観測画像, 統計)
        """
        if not self.config.enabled:
            return slice_img, None
        img = np.asarray(slice_img)
        ref = np.asarray(reference_img)
        if img.shape != ref.shape:
            raise ValueError(f"shape mismatch: observation {img.shape} vs reference {ref.shape}")
        # 観測画像と正解の画像の値の範囲 (0-1 か 0-255) をそろえる
        ref_scaled = ref.astype(np.float64)
        img_f = img.astype(np.float64)
        if img_f.max() > 1.0 + 1e-6 and ref_scaled.max() <= 1.0 + 1e-6:
            ref_scaled = ref_scaled * 255.0
        elif img_f.max() <= 1.0 + 1e-6 and ref_scaled.max() > 1.0 + 1e-6:
            ref_scaled = ref_scaled / 255.0
        palette = _palette_from_reference(ref_scaled)
        n = len(palette)
        observed = np.any(img_f != 0, axis=-1)
        stats = {"observed": int(observed.sum()), "flipped": 0, "num_classes": int(n)}
        if n < 2 or not observed.any():
            return slice_img, stats

        u, r = self._random_fields(key, observed.shape)
        # 各ボクセルの今のラベル = いちばん近い部品の色
        dist = ((img_f[..., None, :] - palette[None, None, :, :]) ** 2).sum(-1)  # (H, W, n)
        cls = dist.argmin(-1)
        flip = observed & (u < self.config.flip_rate)
        new_cls = (cls + 1 + (r % (n - 1))) % n            # 元のラベル以外から一様に選ぶ
        out = img_f.copy()
        out[flip] = palette[new_cls[flip]]
        stats["flipped"] = int(flip.sum())
        return out.astype(img.dtype), stats
