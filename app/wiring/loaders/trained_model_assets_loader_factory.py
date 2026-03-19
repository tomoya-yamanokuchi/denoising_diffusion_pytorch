from __future__ import annotations

from dataclasses import dataclass

from app.wiring.loaders.saved_run_config_loader import SavedRunConfigLoader
from app.wiring.loaders.checkpoint_path_resolver import CheckpointPathResolver
from app.wiring.loaders.conditional_diffusion_assets_loader import (
    ConditionalDiffusionAssetsLoader,
)
from app.wiring.loaders.vaeac_assets_loader import VaeacAssetsLoader


_DIFFUSION_INFER_MODELS = {
    "diffusion",
    "diffusion_1D",
    "conditional_diffusion",
}


@dataclass(frozen=True)
class TrainedModelAssetsLoaderFactory:
    config_loader           : SavedRunConfigLoader
    checkpoint_path_resolver: CheckpointPathResolver

    def create(self, infer_model: str):
        if infer_model == "vaeac":
            return VaeacAssetsLoader()

        if infer_model in _DIFFUSION_INFER_MODELS:
            return ConditionalDiffusionAssetsLoader(
                config_loader            = self.config_loader,
                checkpoint_path_resolver = self.checkpoint_path_resolver,
            )

        raise ValueError(
            f"Unsupported infer_model: {infer_model}. "
            f"Expected one of: 'vaeac', 'diffusion', 'diffusion_1D', 'conditional_diffusion'."
        )
