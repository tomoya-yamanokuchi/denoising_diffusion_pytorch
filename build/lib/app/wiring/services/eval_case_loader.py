from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Any

from omegaconf import DictConfig, OmegaConf

from app.wiring.types.eval_case_config import EvalCaseConfig


def _as_int_tuple(xs: Any) -> tuple[int, ...]:
    """
    start_action_idx を tuple[int,...] に正規化する。
    - YAML list: [1,2,3]
    - OmegaConf ListConfig
    - numpy array など反復可能
    """
    if xs is None:
        return tuple()
    # OmegaConf の ListConfig は普通に反復できる
    return tuple(int(x) for x in xs)


def load_eval_case_configs(cfg_eval: DictConfig) -> List[EvalCaseConfig]:
    """
    cfg.eval から EvalCaseConfig のリストを作る。

    優先順位:
      1) cfg.eval.cases  (新形式: list of {name, dataset_dir, start_action_idx, tags?})
    """
    # ---------- 1) new format: eval.cases ----------
    cases_cfg = OmegaConf.select(cfg_eval, "cases")

    out: List[EvalCaseConfig] = []
    for c in cases_cfg:
        name = str(c.name)
        dataset_dir = Path(str(c.dataset_dir))
        start_action_idx = _as_int_tuple(OmegaConf.select(c, "start_action_idx"))
        tags_raw = OmegaConf.select(c, "tags") or []
        tags = tuple(str(t) for t in tags_raw)

        if not name:
            raise ValueError("eval.cases[].name is empty")
        if not dataset_dir:
            raise ValueError(f"eval.cases[{name}].dataset_dir is empty")
        if len(start_action_idx) == 0:
            raise ValueError(f"eval.cases[{name}].start_action_idx is empty")

        out.append(
            EvalCaseConfig(
                name             = name,
                dataset_dir      = dataset_dir,
                start_action_idx = start_action_idx,
            )
        )

    return out
