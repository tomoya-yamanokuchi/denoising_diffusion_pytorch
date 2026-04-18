

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.arrays import to_torch,to_device


print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    # mode            = "train" # train or eval
    mode            = "test" # train or eva

    image_dim = 64


    # dataset_path    = "./datasets/pidray/train/" # which includes *.png file
    dataset_path    = "/home/haxhi/dataset/denoising_diffusion_pytorch/flower-image/train_concat/"
    # dataset_path    = "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/voxel_images"


    # save_folder     = "./results/x_ray_large"
    # save_folder     = "./results/flower_large_128"
    save_folder     = "./results/flower_large"
    # save_folder     = f"./results/voxel_image_{image_dim}"


    model_idx = 27      # idx model is loaded in test mode



    model = Unet(
        dim = image_dim,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    model.to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size = image_dim,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 500   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = Trainer(
        diffusion,
        dataset_path,
        train_batch_size = 64,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True,              # whether to calculate fid during training
        results_folder = save_folder,
    )





    if mode == "train":
        print(f"mode = train")
        trainer.train()




    elif mode == "test":
        print(f"mode = test")

        import numpy as np
        import os
        from PIL import Image,ImageDraw

        img = Image.new("RGB", (image_dim, image_dim))
        rect_d = ImageDraw.Draw(img)
        rect_d.rectangle(
            [(20, 20), (25, 25)], fill=(255, 1, 1))
        # img.show()
        # rect_d.rectangle(
        #     [(30, 10), (40, 30)], fill=(0, 255, 0))
        img.save("sample.png")
        numpy_image = np.asarray(img)

        normalizer = LimitsNormalizer(numpy_image)
        normalized_cond = normalizer.normalize(numpy_image).transpose(2,0,1)
        normalized_cond = to_torch(normalized_cond)

        cond ={0:{"idx":torch.where(normalized_cond>-1.0),
                    "val":normalized_cond}}

        import ipdb;ipdb.set_trace()
        img = Image.new("RGB", (image_dim, image_dim))
        rect_d = ImageDraw.Draw(img)
        rect_d.rectangle(
            [(30, 50), (35, 64)], fill=(1, 255, 1))
        img.save("sample2.png")
        numpy_image = np.asarray(img)
        normalizer = LimitsNormalizer(numpy_image)
        normalized_cond = normalizer.normalize(numpy_image).transpose(2,0,1)
        normalized_cond = to_torch(normalized_cond)

        cond[1] ={"idx":torch.where(normalized_cond>-1.0),
                    "val":normalized_cond}


        # import ipdb;ipdb.set_trace()

        trainer.load(model_idx)
        sample_image = trainer.model.sample(batch_size=16,return_all_timesteps=True, cond = cond ).detach().cpu()
        batch_size, diffusion_step,image_dim,image_dim,channels = sample_image.shape


        # make denoising process gif
        for i in range(batch_size):
            ims = []
            for j in range(diffusion_step):
                im =  Image.fromarray((torch.permute(sample_image[i][j],(1,2,0))*255.0).numpy().astype(np.uint8))
                ims.append(im.quantize())

            test_save_folder = save_folder +f"/test_{model_idx}_2/"
            os.makedirs(test_save_folder, exist_ok=True)
            save_name = test_save_folder+f"sample_{i}_diffusion.gif"
            ims[0].save(save_name, save_all=True,
                        append_images=ims[1:], optimize=False, duration=50, loop=0)
            ims[-1].save(test_save_folder+f"sample_{i}.png")
            del ims

