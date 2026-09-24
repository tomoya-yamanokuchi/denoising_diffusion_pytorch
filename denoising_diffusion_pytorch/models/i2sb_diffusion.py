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

    def compute_label(
        self,
        step,
        x0,
        xt,
    ):
        """
        Eq. (12) training target.

        label = (x_t - x_0) / sigma_t
        """

        std_fwd = self.get_std_fwd(
            step,
            xdim=x0.shape[1:],
        )

        label = (
            xt - x0
        ) / std_fwd

        return label.detach()

    def compute_pred_x0(
        self,
        step,
        xt,
        net_out,
        clip_denoise=False,
    ):
        """
        Recover x0 from the network output.

        This is the inverse transformation of Eq. (12).
        """

        std_fwd = self.get_std_fwd(
            step,
            xdim=xt.shape[1:],
        )

        pred_x0 = (
            xt
            - std_fwd * net_out
        )

        if clip_denoise:
            pred_x0.clamp_(-1., 1.)

        return pred_x0

    def p_posterior(
        self,
        nprev,
        n,
        x_n,
        x0,
        ot_ode=False,
    ):
        """
        Sample

            p(x_{nprev} | x_n, x_0)

        following the official I2SB implementation.
        """

        assert nprev < n

        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]

        std_delta = (
            std_n**2
            - std_nprev**2
        ).sqrt()

        mu_x0, mu_xn, var = \
            compute_gaussian_product_coef(
                std_nprev,
                std_delta,
            )

        xt_prev = (
            mu_x0 * x0
            + mu_xn * x_n
        )

        if not ot_ode and nprev > 0:
            xt_prev = (
                xt_prev
                + var.sqrt()
                * torch.randn_like(xt_prev)
            )

        return xt_prev

    def ddpm_sampling(
        self,
        steps,
        pred_x0_fn,
        x1,
        cond=None,
        mask=None,
        ot_ode=False,
        log_steps=None,
        verbose=True,
    ):
        """
        Reverse I2SB sampling.

        Starts from X1 and recursively moves toward X0.
        """

        xt = x1.detach().to(
            self.device
        )

        xs = []
        pred_x0s = []

        if log_steps is None:
            log_steps = steps

        assert steps[0] == 0
        assert log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(
            steps[1:],
            steps[:-1],
        )

        if verbose:
            from tqdm import tqdm

            pair_steps = tqdm(
                pair_steps,
                desc="I2SB sampling",
                total=len(steps) - 1,
            )

        for prev_step, step in pair_steps:

            assert prev_step < step

            pred_x0 = pred_x0_fn(
                xt,
                step,
            )

            xt = self.p_posterior(
                prev_step,
                step,
                xt,
                pred_x0,
                ot_ode=ot_ode,
            )


            if cond is not None and mask is not None:

                prev_step_tensor = torch.full(
                    (xt.shape[0],),
                    prev_step,
                    device=self.device,
                    dtype=torch.long,
                )

                xt_observed = self.q_sample(
                    step=prev_step_tensor,
                    x0=cond,
                    x1=x1,
                    ot_ode=ot_ode,
                )

                xt = (
                    (1.0 - mask) * xt_observed
                    + mask * xt
                )

                # import ipdb; ipdb.set_trace()

                obs_error = (
                    (1.0 - mask)
                    * (xt - xt_observed)
                ).abs().max()

                if verbose:
                    print(
                        f"[reinjection] prev_step={prev_step:4d} "
                        f"obs_max_error={obs_error.item():.8e}"
                    )



            if prev_step in log_steps:
                pred_x0s.append(
                    pred_x0.detach().cpu()
                )

                xs.append(
                    xt.detach().cpu()
                )

        def stack_bwd_traj(z):
            return torch.flip(
                torch.stack(
                    z,
                    dim=1,
                ),
                dims=(1,),
            )

        return (
            stack_bwd_traj(xs),
            stack_bwd_traj(pred_x0s),
        )
