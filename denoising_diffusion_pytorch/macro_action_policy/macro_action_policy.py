from .macro_action_policy

class MacroActionPolicy:
    def reset(self, episode_context) -> None:
        ...

    def select(self, obs_z, obs_history, *, last_action, step_idx, env2, save_path):
        """return next_macro_action, aux_infos"""
        ...
