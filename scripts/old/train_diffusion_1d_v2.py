

import torch
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.models.diffusion import GaussianDiffusion
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy,pil_image_load_to_numpy
from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# def normalize_to_neg_one_to_one(img):
#     return img * 2 - 1

# def unnormalize_to_zero_to_one(t):
#     return (t + 1) * 0.5


def get_2d_image_to_1d(image, grid_3_dim , is_shuffle):

        mini_batch_image    = get_2d_image_to_mini_batch_image(image, grid_3_dim, "z")
        mini_batch_dim      = mini_batch_image.shape[0]

        # インデックスを作成 (0から15の範囲)
        indices = torch.tensor([[i, j, k] for i in range(mini_batch_dim) for j in range(mini_batch_dim) for k in range(mini_batch_dim)])  # shape: [4096, 3]
        values  = mini_batch_image[indices[:, 0], indices[:, 1], indices[:, 2]]
        result  = torch.cat((indices/(mini_batch_dim-1.0), values), dim=1)
        # result  = torch.cat((indices, values), dim=1)
        # result  = torch.cat((indices, values), dim=1)

        if is_shuffle is True:
            result_  =result[torch.randperm(result.size(0))]
            result_tp = torch.permute(result_,(1,0))
        else:
            result_tp = torch.permute(result,(1,0))


        return result_tp

def get_2d_image_to_mini_batch_image(image=None, grid_3dim=16, permute = "z"):
    grid_2dim    = image.shape[0]
    grid_3dim    = grid_3dim
    batch_img_len = int(grid_2dim/grid_3dim)

    batch_2d_image_ = torch.zeros((grid_3dim, grid_3dim, grid_3dim, 3))
    k = 0
    for j in range(batch_img_len):
        for i in range(batch_img_len):
            batch_2d_image_[k] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
            k = k+1

    if permute == "z":
        batch_2d_image  = batch_2d_image_
    elif permute == "y":
        batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
    elif permute == "x":
        batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)

    return batch_2d_image


if __name__ == '__main__':

    USER_NAME = "haxhi"
    dataset_path    = "./datasets/flower-image/train_concat/"
    dataset_path    = '/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/dataset_4142435161_13901k/voxel_images_w_multi_color_v1'
    # dataset_path    = "./datasets/pidray/train/"
    # save_folder     = "./results/results_1D_v5_shuffle"
    save_folder     = "./results/results_1D_v7_shuffle"
    # save_folder     = "./results"
    # mode            = "test"
    mode            = "train"
    # model_idx = 20000 # idx model load in training mode
    # model_idx = 95000 # idx model load in training mode
    model_idx = 200000 # idx model load in training mode

    image_size = 64
    grid_3dim  = 16

    dataset = Dataset1D(folder = dataset_path,
                        image_size = image_size,
                        grid_3dim = grid_3dim,
                        is_shuffle = True,
                        )  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=3)
    # aa = next(iter(dataloader))
    # aa = dataset.getitems(0)
    # import ipdb;ipdb.set_trace()
    
    # seq_len = dataset.image_size_flatten/
    # seq_len = dataset.image_size_flatten

    channels, seq_len = dataset.__getitem__(1).shape

    model = Unet1D(
        dim = 128,
        # dim_mults = (1, 2, 4, 8),
        dim_mults = (1, 2, 2, 4, 8),
        channels = channels
    )

    model.to(device)

    # diffusion = GaussianDiffusion1D(
    #     model,
    #     seq_length = seq_len,
    #     timesteps = 1000,
    #     sampling_timesteps = 20,
    #     objective = 'pred_v'
    # )

    diffusion = GaussianDiffusion(
        model,
        seq_length = seq_len,
        timesteps = 1000,
        sampling_timesteps = 20,
        objective = 'pred_v'
    )

    # training_seq = torch.rand(64, channels, seq_len) # features are normalized from 0 to 1
    # loss = diffusion(training_seq)
    # loss.backward()

    # Or using trainer



    # import ipdb;ipdb.set_trace()

    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        results_folder = save_folder,
    )

    if mode == "train":
        trainer.train()

    elif mode == "test":

    # trainer.train()

    # after a lot of training
        trainer.load(model_idx)
        # sample_image = trainer.model.sample(batch_size=16,return_all_timesteps=True).detach().cpu()
        # batch_size, diffusion_step,image_dim,image_dim,channels = sample_image.shape

        # sampled_seq = trainer.model.sample(batch_size = 25).detach().cpu()/

        cond_image_path = "/home/haxhi/workspace/denoising_diffusion_pytorch/0_seq_obs_cast_z_axis0_0.png"
        # cond_image_path = "/home/haxhi/workspace/denoising_diffusion_pytorch/14_seq_obs_cast_z_axis14_0.png"
        # cond_image_path = "/home/haxhi/workspace/denoising_diffusion_pytorch/oracle_obs_cast_z_axis0.png"
        # cond_image_path2 = "/home/haxhi/workspace/denoising_diffusion_pytorch/0_seq_obs_cast_x_axis0_0.png"
        

        cond_image      =  pil_image_load_to_numpy(cond_image_path)
        cond_image_1d_  = get_2d_image_to_1d(image=to_torch(cond_image), grid_3_dim=grid_3dim, is_shuffle=False)
        cond_image_1d   = to_np(cond_image_1d_)

        # cond_image_1d_tmp = get_2d_image_to_1d(image=to_torch(pil_image_load_to_numpy(cond_image_path2)), grid_3_dim=grid_3dim, is_shuffle=True)[None,:,:]
        # cond_image_1d_tmp = torch.ones(1,6,4096)+torch.rand(1,6,4096)
        # cond_image_1d_tmp = torch.rand(1,6,4096)
        
        # cond_image_1d_tmp = cond_image_1d_tmp_.clone()
        # cond_image_1d_tmp = cond_image_1d.clone()[None,:,:]

        # import ipdb;ipdb.set_trace()
        # cond_image_1d     =  cond_image_1d.transpose(1,0)
        # cond_image_1d_tmp =  cond_image_1d_tmp.transpose(1,0)
        normalizer_values      = LimitsNormalizer(cond_image_1d[3:,:])
        normalizer_indices     = LimitsNormalizer(cond_image_1d[:3,:])

        voxel_indices = to_torch(normalizer_indices.normalize(cond_image_1d[:3,:]))
        voxel_values  = to_torch(normalizer_values.normalize(cond_image_1d[3:,:]))

        # cond_image_1d = normalize_to_neg_one_to_one(cond_image_1d)
        # voxel_indices = to_torch(normalize_to_neg_one_to_one(cond_image_1d[:3,:]))
        # voxel_values  = to_torch(normalize_to_neg_one_to_one(cond_image_1d[3:,:]))

        # voxel_indices = to_torch(cond_image_1d[:3,:])
        # voxel_values  = to_torch(cond_image_1d[3:,:])

        # import ipdb;ipdb.set_trace()

        cond = {0:{ "idx":torch.where(voxel_values.mean(0)>-1.0),
                    "val":voxel_values,
                    "pos":voxel_indices,
                    "data":torch.cat((voxel_indices,voxel_values), dim=0)}}

        # import ipdb;ipdb.set_trace()

        # # xx = torch.zeros(5,6,4096)
        # cond_image_1d_tmp =  cond_image_1d_tmp.repeat(10,1,1)
        # # cond_image_1d_tmp[:,3:,:] = 0.5
        # # import ipdb;ipdb.set_trace()

        # for cond_num ,items in cond.items():
        #     cond_len = items["idx"][0].shape[0]
        #     # import ipdb;ipdb.set_trace()
            
        # #     # xx[:,items["idx"][0],items["idx"][1],items["idx"][2]] = items["val"][items["idx"]].clone()
        # #     # cond_image_1d_tmp[:,items["idx"][0],items["idx"][1]] = items["val"][items["idx"],].clone()
        # #     # cond_image_1d_tmp[:,items["idx"][0]+2,items["idx"][1]] = items["val"][items["idx"][0],items["idx"][1]].clone()
        #     # cond_image_1d_tmp[:,3:,items["idx"][0]] =  items["val"][:,items["idx"][0]].clone()
        #     cond_size = cond_image_1d_tmp[:,3:,items["idx"][0]].shape


        #     cond_image_1d_tmp[:,3:,items["idx"][0]] =  items["val"][:,items["idx"][0]].clone()
        #     cond_image_1d_tmp[:,:3,items["idx"][0]] =  items["pos"][:,items["idx"][0]].clone()

        #     # aa =  items["pos"][:,items["idx"][0]].clone()
        #     # bb = torch.roll(aa,-1,0)
        #     # cond_image_1d_tmp[:,:3,items["idx"][0]] =  bb

        #     # cond_image_1d_tmp[0,3:,:cond_len]= items["val"][:,items["idx"][0]].clone()
        #     # cond_image_1d_tmp[0,3:,:cond_len]= 0.9
        #     # cond_image_1d_tmp[0,:3,:cond_len]= items["pos"][:,items["idx"][0]].clone()
        #     # import ipdb;ipdb.set_trace()
            
        #     # cond_image_1d_tmp[:,3:,:cond_len]= items["val"][:,items["idx"][0]].clone().repeat(10,1,1)
        #     # cond_image_1d_tmp[:,:3,:cond_len]= items["pos"][:,items["idx"][0]].clone().repeat(10,1,1)
        #     # import ipdb;ipdb.set_trace()

        #     # cond_image_1d_tmp[:,3:,:cond_len]= items["data"][3:,items["idx"][0]].clone()
        #     # cond_image_1d_tmp[:,:3,:cond_len]= items["data"][:3,items["idx"][0]].clone()
            
        #     # cond_image_1d_tmp[:,3:,:cond_len]= items["data"][3:,items["idx"][0]].detach().clone()
        #     # cond_image_1d_tmp[:,:3,:cond_len]= items["data"][:3,items["idx"][0]].detach().clone()
            
        #     # cond_image_1d_tmp[:,:,:cond_len]= items["data"][:,items["idx"][0]].clone()
            

        #     # import ipdb;ipdb.set_trace()
            
        # # return x



        # import ipdb;ipdb.set_trace()

        sampled_seq = trainer.model.sample(batch_size = 10, return_all_timesteps=True, cond = cond)
        # sampled_seq = trainer.model.sample(batch_size = 21, return_all_timesteps=False, cond = cond)

        # sampled_seq = to_device(cond_image_1d_tmp)
        # import ipdb;ipdb.set_trace()
        dd = sampled_seq[0]
        
        # sampled_image = trainer.get_1d_to_2d_images(sampled_seq).detach().cpu()
        sampled_image = trainer.get_1d_to_2d_images(dd).detach().cpu()
        # sampled_image = trainer.get_1d_to_2d_images(sampled_seq[0].squeeze()).detach().cpu()

        import numpy as np
        import os
        from PIL import Image

        # sampled_image = to_np(torch.permute(sampled_image,(0,2,3,1)))
        # sampled_image = torch.permute(sampled_image,(0,2,3,1))
        diffusion_step,image_dim,image_dim,channels = sampled_image.shape
        ims = []
        for j in range(diffusion_step):
            # import ipdb;ipdb.set_trace()
            im =  Image.fromarray((torch.permute(sampled_image[j],(1,2,0))*255.0).numpy().astype(np.uint8))
            ims.append(im)

        ims[0].save("hoge.gif", save_all=True,
                    append_images=ims[1:], optimize=False, duration=100,)

        # import ipdb;ipdb.set_trace()

        # pil_image_save_from_numpy(sampled_image[0],f"./hoge_{3}.png")
        pil_image_save_from_numpy(to_np(torch.permute(sampled_image,(0,2,3,1)))[-1],f"./hoge_{3}.png")


        # sampled_seq.shape # (4, 32, 128)

        # import ipdb;ipdb.set_trace()




















    # model = Unet(
    #     dim = 64,
    #     dim_mults = (1, 2, 4, 8),
    #     flash_attn = True
    # )

    # model.to(device)

    # diffusion = GaussianDiffusion(
    #     model,
    #     image_size = 32,
    #     timesteps = 1000,           # number of steps
    #     sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    # )

    # trainer = Trainer(
    #     diffusion,
    #     dataset_path,
    #     train_batch_size = 32,
    #     train_lr = 8e-5,
    #     train_num_steps = 700000,         # total training steps
    #     gradient_accumulate_every = 2,    # gradient accumulation steps
    #     ema_decay = 0.995,                # exponential moving average decay
    #     amp = True,                       # turn on mixed precision
    #     calculate_fid = True,              # whether to calculate fid during training
    #     results_folder = save_folder,
    # )


    # if mode == "train":
    #     trainer.train()

    # elif mode == "test":

    #     import numpy as np
    #     import os
    #     from PIL import Image


    #     model_idx = 35
    #     trainer.load(model_idx)
    #     sample_image = trainer.model.sample(batch_size=8,return_all_timesteps=True).detach().cpu()
    #     batch_size, diffusion_step,image_dim,image_dim,channels = sample_image.shape

    #     for  i in range(batch_size):
    #         ims = []
    #         for j in range(diffusion_step):
    #             # import ipdb;ipdb.set_trace()
    #             im =  Image.fromarray((torch.permute(sample_image[i][j],(1,2,0))*255.0).numpy().astype(np.uint8))
    #             ims.append(im)

    #         test_save_folder = save_folder +f"/test_{model_idx}/"
    #         os.makedirs(test_save_folder, exist_ok=True)
    #         save_name = test_save_folder+f"sample_{i}_diffusion.gif"
    #         ims[0].save(save_name, save_all=True,
    #                     append_images=ims[1:], optimize=False, duration=50, loop=0)
    #         del ims


    #     import ipdb;ipdb.set_trace()