# denoising_diffusion_pytorch/models/i2sb_diffusion.py

from functools import partial

import numpy as np
import torch


def unsqueeze_xdim(z, xdim):
    """
    Reshape [B] or scalar timestep-dependent values
    so that they can be broadcast to [B, C, H, W].
    """
    return z.reshape(*z.shape, *((1,) * len(xdim)))


def compute_gaussian_product_coef(sigma1, sigma2):
    """
    Given

        p1 = N(x_t | x_0, sigma1^2)
        p2 = N(x_t | x_1, sigma2^2)

    return

        p1 * p2
        = N(
            x_t |
            coef1 * x0 + coef2 * x1,
            var
        )
    """

    denom = sigma1**2 + sigma2**2

    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom

    var = (
        sigma1**2
        * sigma2**2
        / denom
    )

    return coef1, coef2, var


class I2SBDiffusion:

    def __init__(
        self,
        betas,
        device,
    ):

        self.device = device

        # --------------------------------------------
        # Analytic standard deviations
        # Eq. (11) in the I2SB paper
        # --------------------------------------------

        std_fwd = np.sqrt(
            np.cumsum(betas)
        )

        std_bwd = np.sqrt(
            np.flip(
                np.cumsum(
                    np.flip(betas)
                )
            )
        )

        mu_x0, mu_x1, var = \
            compute_gaussian_product_coef(
                std_fwd,
                std_bwd,
            )

        std_sb = np.sqrt(var)

        # --------------------------------------------
        # Convert to torch tensors
        # --------------------------------------------

        to_torch = partial(
            torch.tensor,
            dtype=torch.float32,
        )

        self.betas = (
            to_torch(betas)
            .to(device)
        )

        self.std_fwd = (
            to_torch(std_fwd)
            .to(device)
        )

        self.std_bwd = (
            to_torch(std_bwd)
            .to(device)
        )

        self.std_sb = (
            to_torch(std_sb)
            .to(device)
        )

        self.mu_x0 = (
            to_torch(mu_x0)
            .to(device)
        )

        self.mu_x1 = (
            to_torch(mu_x1)
            .to(device)
        )

    def get_std_fwd(
        self,
        step,
        xdim=None,
    ):

        std_fwd = self.std_fwd[step]

        if xdim is None:
            return std_fwd

        return unsqueeze_xdim(
            std_fwd,
            xdim,
        )

    def q_sample(
        self,
        step,
        x0,
        x1,
        ot_ode=False,
    ):
        """
        Sample

            q(x_t | x_0, x_1)

        according to Eq. (11).
        """

        assert x0.shape == x1.shape

        _, *xdim = x0.shape

        mu_x0 = unsqueeze_xdim(
            self.mu_x0[step],
            xdim,
        )

        mu_x1 = unsqueeze_xdim(
            self.mu_x1[step],
            xdim,
        )

        std_sb = unsqueeze_xdim(
            self.std_sb[step],
            xdim,
        )

        xt = (
            mu_x0 * x0
            + mu_x1 * x1
        )

        if not ot_ode:
            xt = (
                xt
                + std_sb
                * torch.randn_like(xt)
            )

        return xt.detach()
