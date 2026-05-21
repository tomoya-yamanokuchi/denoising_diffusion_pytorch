from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from denoising_diffusion_pytorch.utils.os_utils import pickle_utils
from ..types import EpisodeContext, EpisodeResult


@dataclass
class EpisodeResultWriter:
    def save(
        self,
        episode_ctx   : EpisodeContext,
        episode_result: EpisodeResult,
    ) -> None:
        save_root = str(episode_ctx.path.artifact_episodic_root)

        rollout_data = {
            "observations"          : np.asarray(episode_result.observations),
            "actions"               : np.asarray(episode_result.actions),
            "planned_actions"       : np.asarray(episode_result.planned_actions),
            "executed_actions"      : np.asarray(episode_result.executed_actions),

            "cutting_error_volumes" : np.asarray(episode_result.cutting_error_volumes),

            # backward-compatible / diagnostic
            "infos"                 : np.asarray(episode_result.infos),

            # paper metric
            "part_remaining_rates"  : np.asarray(episode_result.part_remaining_rates),
            "part_occupancy_rates"  : np.asarray(episode_result.part_occupancy_rates),

            "execution_error_infos" : episode_result.execution_error_infos,
        }
        pickle_utils().save(
            dataset=rollout_data,
            save_path=f"{save_root}/rollout_data.pickle",
        )

        visualization_data = {
            "observations"        : np.asarray(episode_result.observations),
            "actions"             : np.asarray(episode_result.actions),
            "intermediate_actions": episode_result.intermediate_actions,

            "planned_actions"     : np.asarray(episode_result.planned_actions),
            "executed_actions"    : np.asarray(episode_result.executed_actions),
            "planned_intermediate_actions" : episode_result.planned_intermediate_actions,
            "executed_intermediate_actions": episode_result.executed_intermediate_actions,
            "execution_error_infos": episode_result.execution_error_infos,
        }
        pickle_utils().save(
            dataset   = visualization_data,
            save_path = f"{save_root}/visualization_data.pickle",
        )
