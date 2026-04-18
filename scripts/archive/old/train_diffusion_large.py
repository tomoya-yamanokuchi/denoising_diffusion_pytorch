

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    # dataset_path    = "./datasets/flower-image/train_concat/"
    dataset_path    = "./datasets/pidray/train/"
    # save_folder     = "./results_xray"
    save_folder     = "./results/x_ray_large"
    mode            = "train" # train or eval
    model_idx = 27 # idx model load in training mode

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    model.to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = Trainer(
        diffusion,
        dataset_path,
        train_batch_size = 128,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True,              # whether to calculate fid during training
        results_folder = save_folder,
    )


    if mode == "train":
        trainer.train()

    elif mode == "test":

        import numpy as np
        import os
        from PIL import Image



        trainer.load(model_idx)
        sample_image = trainer.model.sample(batch_size=16,return_all_timesteps=True).detach().cpu()
        batch_size, diffusion_step,image_dim,image_dim,channels = sample_image.shape

        for  i in range(batch_size):
            ims = []
            for j in range(diffusion_step):
                # import ipdb;ipdb.set_trace()
                im =  Image.fromarray((torch.permute(sample_image[i][j],(1,2,0))*255.0).numpy().astype(np.uint8))
                ims.append(im.quantize())

            test_save_folder = save_folder +f"/test_{model_idx}/"
            os.makedirs(test_save_folder, exist_ok=True)
            save_name = test_save_folder+f"sample_{i}_diffusion.gif"
            ims[0].save(save_name, save_all=True,
                        append_images=ims[1:], optimize=False, duration=50, loop=0)
            del ims


        import ipdb;ipdb.set_trace()