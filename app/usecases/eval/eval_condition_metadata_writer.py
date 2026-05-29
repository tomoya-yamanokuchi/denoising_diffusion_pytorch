from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ipdb
import yaml
from omegaconf import DictConfig, OmegaConf


@dataclass
class EvalConditionMetadataWriter:
    """
    Save run-level evaluation metadata.

    This metadata describes the experimental condition that produced
    the rollout_data.pickle files under artifact_static_root.

    Expected output:
        <artifact_static_root>/condition_metadata.yaml
    """

    filename: str = "condition_metadata.yaml"

    def save(
        self,
        cfg: DictConfig,
        artifact_static_root: str | Path,
    ) -> Path:
        root = Path(artifact_static_root)
        root.mkdir(parents=True, exist_ok=True)

        metadata = self._build_metadata(
            cfg=cfg,
            artifact_static_root=root,
        )

        save_path = root / self.filename
        with save_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                metadata,
                f,
                sort_keys=False,
                allow_unicode=True,
            )

        return save_path

    def _build_metadata(
        self,
        cfg: DictConfig,
        artifact_static_root: Path,
    ) -> dict[str, Any]:
        eta = self._get_cutting_risk_threshold(cfg)
        delta = self._get_execution_error_delta(cfg)
        inference = self._get_policy_inference_metadata(cfg)

        return {
            "condition": self._build_condition_name(
                eta=eta,
                delta=delta,
                guidance_scale=inference["guidance_scale"],
                sample_image_num=inference["sample_image_num"],
                sampling_timesteps=inference["sampling_timesteps"],
            ),
            "artifact_static_root": str(artifact_static_root),

            # Paper / experiment condition
            "eta": eta,
            "delta": delta,
            "cutting_risk_threshold": eta,

            "guidance_scale": inference["guidance_scale"],
            "sample_image_num": inference["sample_image_num"],
            "sampling_timesteps": inference["sampling_timesteps"],

            "execution_error": {
                "enabled"      : bool(self._cfg_get(cfg.eval.execution_error, "enabled", False)),
                "mode"         : str(self._cfg_get(cfg.eval.execution_error, "mode", "none")),
                "max_abs_shift": delta,
                "seed"         : self._cfg_get(cfg.eval.execution_error, "seed", None),
            },

            # Useful run information
            "eval": {
                "cases_name": str(self._cfg_get(cfg.eval.cases, "name", "")),
                "case_names": [
                    str(case.name) for case in self._cfg_get(cfg.eval.cases, "cases", [])
                ],
                "num_episodes": int(self._cfg_get(cfg.eval.task, "num_episodes", -1)),
                "task_step": int(self._cfg_get(cfg.eval.task, "task_step", -1)),
                "train_run_dir": str(self._cfg_get(cfg.eval, "train_run_dir", "")),
                "epoch": str(self._cfg_get(cfg.eval, "epoch", "")),
                "infer_model": str(self._cfg_get(cfg.eval.policy, "infer_model", "")),
            },

            "policy": {
                "control_mode": str(self._cfg_get(cfg.eval.policy.control, "mode", "")),
                "decision_param": self._to_plain_container(
                    self._cfg_get(cfg.eval.policy.decision, "param", {})
                ),
                "inference": inference,
            },
            "experiment_tag": self._cfg_get(cfg.log, "tag", None),
            # Full resolved config snapshot.
            # This makes the result self-contained for later analysis.
            "resolved_config": self._to_plain_container(cfg),
        }

    def _get_cutting_risk_threshold(self, cfg: DictConfig) -> float:
        # In the current implementation, η corresponds to ucb_lb.
        return float(cfg.eval.policy.decision.param.ucb_lb)

    def _get_execution_error_delta(self, cfg: DictConfig) -> int:
        execution_error_cfg = self._cfg_get(cfg.eval, "execution_error", None)
        if execution_error_cfg is None:
            return 0

        return int(self._cfg_get(execution_error_cfg, "max_abs_shift", 0))

    def _get_policy_inference_metadata(self, cfg: DictConfig) -> dict[str, Any]:
        inference_cfg = self._cfg_get(cfg.eval.policy, "inference", None)

        return {
            "guidance_scale": float(
                self._cfg_get(inference_cfg, "guidance_scale", 0.2)
            ),
            "sample_image_num": int(
                self._cfg_get(inference_cfg, "sample_image_num", 32)
            ),
            "sampling_timesteps": int(
                self._cfg_get(inference_cfg, "sampling_timesteps", -1)
            ),
        }


    def _build_condition_name(
        self,
        eta: float,
        delta: int,
        guidance_scale: float,
        sample_image_num: int,
        sampling_timesteps: int,
    ) -> str:
        eta_label = self._format_float_label(eta)
        w_label = self._format_float_label(guidance_scale)

        return (
            f"eta_{eta_label}"
            f"_delta_{delta}"
            f"_w_{w_label}"
            f"_M_{sample_image_num}"
            f"_S_{sampling_timesteps}"
        )

    def _format_float_label(self, value: float) -> str:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        if "." not in text:
            text = f"{text}.0"
        return text.replace(".", "p").replace("-", "m")

    def _to_plain_container(self, value: Any) -> Any:
        if isinstance(value, DictConfig):
            return OmegaConf.to_container(value, resolve=True)

        return value

    def _cfg_get(self, cfg: Any, key: str, default: Any) -> Any:
        if cfg is None:
            return default

        if isinstance(cfg, dict):
            return cfg.get(key, default)

        return getattr(cfg, key, default)
