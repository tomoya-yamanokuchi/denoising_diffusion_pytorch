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

            "step_normalized_cutting_error_rates": np.asarray(
                episode_result.step_normalized_cutting_error_rates
            ),
            "episode_cumulative_normalized_cutting_error_rate": float(
                np.sum(episode_result.step_normalized_cutting_error_rates)
            ),
            "oracle_target_shape_vol": float(
                np.asarray(episode_result.oracle_target_shape_vols)[-1]
            ),

            # paper metric
            "part_remaining_rates"  : np.asarray(episode_result.part_remaining_rates),
            "part_occupancy_rates"  : np.asarray(episode_result.part_occupancy_rates),

            "execution_error_infos" : episode_result.execution_error_infos,
        }

        self._add_mask_data_if_available(
            rollout_data=rollout_data,
            episode_result=episode_result,
        )

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


    def _add_mask_data_if_available(
        self,
        rollout_data: dict,
        episode_result: EpisodeResult,
    ) -> None:
        """Append optional voxel masks for post-hoc overcut visualization."""
        if episode_result.oracle_target_mask is not None:
            rollout_data["oracle_target_mask"] = np.asarray(
                episode_result.oracle_target_mask,
                dtype=bool,
            )

        if episode_result.step_cutting_error_masks is not None:
            rollout_data["step_cutting_error_masks"] = np.asarray(
                episode_result.step_cutting_error_masks,
                dtype=bool,
            )

        if episode_result.cumulative_cutting_error_masks is not None:
            rollout_data["cumulative_cutting_error_masks"] = np.asarray(
                episode_result.cumulative_cutting_error_masks,
                dtype=bool,
            )

        if episode_result.final_cutting_error_mask is not None:
            rollout_data["final_cutting_error_mask"] = np.asarray(
                episode_result.final_cutting_error_mask,
                dtype=bool,
            )
