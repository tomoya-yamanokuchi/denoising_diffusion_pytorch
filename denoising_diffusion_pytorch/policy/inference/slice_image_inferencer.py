from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from denoising_diffusion_pytorch.policy.types import PlanningPolicyInput


class SliceImageInferencer(ABC):
    @abstractmethod
    def predict(self, planning_input: PlanningPolicyInput) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Last-step generated images with shape:
            (batch, height, width, channel)
        """
        raise NotImplementedError
