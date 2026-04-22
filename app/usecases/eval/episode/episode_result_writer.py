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
            "observations"       : np.asarray(episode_result.observations),
            "actions"            : np.asarray(episode_result.actions),
            "rewards"            : np.asarray(episode_result.rewards),
            "infos"              : np.asarray(episode_result.infos),
            "removal_performance": np.asarray(episode_result.removal_performance),
        }
        pickle_utils().save(
            dataset=rollout_data,
            save_path=f"{save_root}/rollout_data.pickle",
        )

        visualization_data = {
            "observations"        : np.asarray(episode_result.observations),
            "actions"             : np.asarray(episode_result.actions),
            "intermediate_actions": episode_result.intermediate_actions,
        }
        pickle_utils().save(
            dataset   = visualization_data,
            save_path = f"{save_root}/visualization_data.pickle",
        )
