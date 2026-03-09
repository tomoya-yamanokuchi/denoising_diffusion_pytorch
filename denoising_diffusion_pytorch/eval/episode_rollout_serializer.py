import numpy as np
from .types import EpisodeRolloutSnapshot


class EpisodeRolloutSerializer:
    def to_rollout_payload(self, snapshot: EpisodeRolloutSnapshot) -> dict:
        steps = snapshot.steps
        return {
            "observations"       : np.asarray([s.obs_z for s in steps]),
            "actions"            : np.asarray([s.last_action for s in steps], dtype=np.int64),
            "rewards"            : np.asarray([s.reward for s in steps], dtype=np.float32),
            "infos"              : np.asarray([s.target_removal_rate for s in steps], dtype=np.float32),
            "removal_performance": np.asarray([s.removal_performance for s in steps], dtype=np.float32),
        }

    def to_visualization_payload(self, snapshot: EpisodeRolloutSnapshot) -> dict:
        steps = snapshot.steps
        return {
            "observations"        : np.asarray([s.obs_z for s in steps]),
            "actions"             : np.asarray([s.last_action for s in steps], dtype=np.int64),
            "intermediate_actions": [list(s.macro_action) for s in steps],
        }
