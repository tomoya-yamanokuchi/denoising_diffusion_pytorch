from dataclasses import dataclass
from pathlib import Path
from typing import Any

from denoising_diffusion_pytorch.eval.episode_runner import EpisodeRunner



@dataclass(frozen=True)
class EvalContext:
    cfg                : Any
    run_dir            : Path
    eval_cases         : Any
    mesh_factory       : Any
    case_ctx_factory   : Any
    episode_ctx_factory: Any
    episode_runner     : "EpisodeRunner"
    orchestrator       : ""
    # + policy(method) があるならここに
