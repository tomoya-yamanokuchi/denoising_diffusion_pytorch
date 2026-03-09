from .macro_action_policy import MacroActionPolicy


class WarmStartThenPolicy(MacroActionPolicy):
    def __init__(self,
            warm_start_action,
            delegate_policy,
            split_obs_updater=None,
        ):
        self._warm_start_action = warm_start_action
        self.delegate_policy    = delegate_policy
        self._used              = False
        self._split_obs_updater = split_obs_updater

    def reset(self, episode_context):
        self._used = False
        self.delegate_policy.reset(episode_context)

    def select(self, obs_z, obs_history, *, last_action, step_idx, env2, save_path):
        if not self._used:
            self._used = True
            if self._split_obs_updater is not None:
                self._split_obs_updater(self._warm_start_action)  # policy.update_split_obs_config 相当
            return self._warm_start_action, {"mode": "warm_start"}
        return self._delegate.select(obs_z, obs_history, last_action=last_action, step_idx=step_idx, env2=env2, save_path=save_path)
