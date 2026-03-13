from __future__ import annotations

from dataclasses import dataclass

from denoising_diffusion_pytorch.utils.serialization import load_diffusion, load_vaeac

from app.wiring.types.policy_model_assets import PolicyModelAssets


@dataclass
class PolicyModelAssetsLoader:
    def load(self, eval_cfg) -> PolicyModelAssets:
        infer_model = eval_cfg.policy_config.infer_model

        if infer_model == "vaeac":
            experiment = load_vaeac(eval_cfg.loadpath, epoch=eval_cfg.epoch)
        elif infer_model in {"diffusion", "diffusion_1D", "conditional_diffusion"}:
            experiment = load_diffusion(eval_cfg.loadpath, epoch=eval_cfg.epoch)
        else:
            raise ValueError(
                f"Unsupported infer_model for policy model assets loading: {infer_model}"
            )

        import ipdb; ipdb.set_trace()

        return PolicyModelAssets(
            diffusion = experiment.ema,
            dataset   = experiment.dataset,
            trainer   = experiment.trainer,
        )
