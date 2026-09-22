import numpy as np
import torch

from denoising_diffusion_pytorch.models.i2sb_diffusion \
    import I2SBDiffusion


def make_beta_schedule(
    n_timestep=1000,
    linear_start=1e-4,
    linear_end=2e-2,
):
    betas = (
        torch.linspace(
            linear_start ** 0.5,
            linear_end ** 0.5,
            n_timestep,
            dtype=torch.float64,
        ) ** 2
    )

    return betas.numpy()


interval = 1000
beta_max = 0.3

betas = make_beta_schedule(
    n_timestep=interval,
    linear_end=beta_max / interval,
)

betas = np.concatenate([
    betas[:interval // 2],
    np.flip(
        betas[:interval // 2]
    ),
])


diffusion = I2SBDiffusion(
    betas=betas,
    device="cpu",
)


for step in [
    0,
    250,
    500,
    750,
    999,
]:
    print(
        step,
        "mu_x0 =",
        diffusion.mu_x0[step].item(),
        "mu_x1 =",
        diffusion.mu_x1[step].item(),
        "std_sb =",
        diffusion.std_sb[step].item(),
    )




print("\n--- Step 2 sanity check ---")

x0 = torch.randn(
    2, 3, 16, 16
)

x1 = torch.randn(
    2, 3, 16, 16
)

step = torch.tensor([
    250,
    750,
])

xt = diffusion.q_sample(
    step=step,
    x0=x0,
    x1=x1,
    ot_ode=False,
)

label = diffusion.compute_label(
    step=step,
    x0=x0,
    xt=xt,
)

pred_x0 = diffusion.compute_pred_x0(
    step=step,
    xt=xt,
    net_out=label,
)

error = (
    pred_x0 - x0
).abs().max()

print(
    "max reconstruction error =",
    error.item()
)
