

import os
from tqdm import tqdm
from multiprocessing import cpu_count
import pickle
import torch
from torch import nn
import yaml
import random

import numpy as np
import os
from PIL import Image,ImageDraw
import pyvista as pv

from denoising_diffusion_pytorch.utils.config import Config

from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env

from denoising_diffusion_pytorch.utils.setup import Parser
from denoising_diffusion_pytorch.utils.benchmark_model_utils import get_benchmark_model

from denoising_diffusion_pytorch.utils.serialization import load_diffusion,load_vaeac


from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy



# from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml

# from denoising_diffusion_pytorch.utils.parser import Parser  # 例：もとの定義に合わせる



print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"




if __name__ == '__main__':

    #---------------------------------- setup ----------------------------------#
    # load_path = '/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_H100_real_models_dataset_v1_7_tmp'
    # load_path = '/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D344_H100_real_models_dataset_v2_1'
    # load_path = '/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac/D344_H100_real_models_dataset_v2_2'
    
    # load_path = '/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/conditional_diffusion/T1000_D344_H100_real_models_dataset_v2_1'
    # load_path = '/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/conditional_diffusion/T1000_D344_H100_real_models_dataset_v2_2'
    # load_path = '/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/conditional_diffusion2/T1000_D64_4090b_real_models_dataset_v2_tmp18_2'
    # load_path = '/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/conditional_diffusion2/T1000_D64_dataset_13901k_v11/'
    load_path = '/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/conditional_diffusion2/T1000_D344_H100_real_models_dataset_v2_18'
    

    
    print(load_path)

    #---------------------------------- loading ----------------------------------#

    # import ipdb;ipdb.set_trace()

    ## diffusion model load
    diffusion_experiment    = load_diffusion(load_path)
    # diffusion_experiment    = load_vaeac(load_path,epoch=74000)
    # diffusion               = diffusion_experiment.ema
    # dataset                 = diffusion_experiment.dataset
    trainer                 = diffusion_experiment.trainer

    trainer.save_and_sample_every = 2000

    # from torchvision import transforms as T, utils
    # import math


    # num_samples = 9
    # milestone = 1

    # all_target_list =[]
    # all_images_list = []
    # all_masks_list = []
    
    # data = next(trainer.val_dl)
    # trainer.omega = 10
    # for n in range(num_samples):
    #     img  = data["image"]
    #     mask = data["observed"].to(trainer.device)


    #     # images = self.ema.ema_model.sample(batch_size=n)
        
    #     images = trainer.ema.ema_model.sample(batch_size=1,mask=mask)
    #     # images = trainer.ema.model.sample(batch_size=1,mask=mask)
    #     # images = trainer.model.sample(batch_size=1,mask=mask)
        
        

    #     all_images_list.append(images)
    #     all_masks_list.append(mask)
    #     all_target_list.append(img)

    # all_images = torch.cat(all_images_list, dim=0)  # shape: [N, C, H, W]
    # all_masks = torch.cat(all_masks_list, dim=0)    # shape: [N, c, H, W]
    # all_targets = torch.cat(all_target_list, dim=0)  # shape: [N, C, H, W]

    # save_dir = '/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D'
    # utils.save_image(all_images, (save_dir+f'sample-{milestone}_pred.png'), nrow = int(math.sqrt(num_samples)))
    # utils.save_image(all_masks,  (save_dir+f'sample-{milestone}_mask.png'), nrow = int(math.sqrt(num_samples)))
    # utils.save_image(all_targets,(save_dir+f'sample-{milestone}_target.png'), nrow = int(math.sqrt(num_samples)))







    # import ipdb;ipdb.set_trace()
    
    torch.cuda.empty_cache() 

    trainer.train()

