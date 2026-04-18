

import os

import torch
from torch import nn

from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.serialization import load_diffusion


from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np


from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils

class Parser(parser):
    dataset: str = 'Image_diffusion_2D'
    config: str =  'config.vae'

#---------------------------------- setup ----------------------------------#


args = Parser().parse_args('diffusion_plan')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#


#---------------------------------- loading ----------------------------------#
## diffusion model load
diffusion_experiment    = load_diffusion(args.diffusion_loadpath, epoch=args.diffusion_epoch)
diffusion               = diffusion_experiment.ema
dataset                 = diffusion_experiment.dataset
# renderer                = diffusion_experiment.renderer


import ipdb;ipdb.set_trace()



import numpy as np
import os
from PIL import Image,ImageDraw
import pyvista as pv


dataset_path                =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/Boxy_99"
mesh_source                 = f"{dataset_path}/blend/"
sample_image_num            = args.batch_size
test_save_folder            = args.savepath
cond_save_path              = os.path.normpath(f"{test_save_folder}/voxel_images")
create_folder(cond_save_path)


## hard
# slice_tag         = [3,4,5,6,7,8,9,10,11,12,14]

##  mid
# slice_tag         = [3,6,9,10,14]

## soft
# slice_tag         = [3,9,10]

## intermediate
# slice_tag         = [0,2,3,6,7,9,11,13,15]

## ??
# slice_tag         = [0,1,4,5,8,10,12,14,15]

## composite_1
# slice_tag         = [0,3,4,7,8,11,12,15]
## composite_2
# slice_tag         = [0,2,3,6,7,9,13,15]





## load inner boxes and merge
path, f_name = get_path(mesh_source,".stl")
mesh2 = pv.read(path[1])
mesh3 = pv.read(path[2])
mesh4 = pv.read(path[3])
merged = mesh2.merge(mesh3)
merged = merged.merge(mesh4)


## voxelize mesh and get sliced image
s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
                    "side_length":16}


box_array_handler   = pv_box_array(grid_config=s_grid_config)
_                   = box_array_handler.cast_mesh_to_box_array(mesh=merged)
box_arrays_data     = box_array_handler.get_box_array_data()


nearby_cells = box_arrays_data.boxes
colors       = box_arrays_data.colors
centers      = box_arrays_data.grid_centers
grid_2dim    = box_arrays_data.grid_2dim_size
grid_3dim    = box_arrays_data.grid_3dim_size
batch_image_map = box_array_handler.batch_image_map


## save image
imgs_z = box_array_handler.get_box_color_to_2d_image(box_color=colors,permute="z")
pil_image_ = Image.fromarray((imgs_z*255).astype(np.uint8))
save_name = f"{cond_save_path}/cast_z_axis{0}.png"
pil_image_.save(save_name)


# slice_img = box_array_handler.get_slice_image(image=imgs_z,slice_tag=[3,4,6,12,14])
slice_img = box_array_handler.get_slice_image(image=imgs_z,slice_tag=slice_tag)
pil_image = Image.fromarray((slice_img*255).astype(np.uint8))
save_name = f"{cond_save_path}/cast_z_axis_cond{0}.png"
pil_image.save(save_name)


img = slice_img
numpy_image = np.asarray(img)

normalizer = LimitsNormalizer(numpy_image)
normalized_cond = normalizer.normalize(numpy_image).transpose(2,0,1)
normalized_cond = to_torch(normalized_cond)

cond ={0:{"idx":torch.where(normalized_cond>-1.0),
            "val":normalized_cond}}


# trainer.load(model_idx)?
sample_image = diffusion.model.sample(batch_size=sample_image_num,return_all_timesteps=True, cond = cond ).detach().cpu()
batch_size, diffusion_step,image_dim,image_dim,channels = sample_image.shape


# make denoising process gif
batch_images = (torch.permute(sample_image,(0,1,3,4,2))*255.0).numpy().astype(np.uint8)
for i in range(batch_size):
    img_save_path = os.path.normpath(f"{test_save_folder}/batch_{i}")
    create_folder(img_save_path)

    ims = []
    for j in range(diffusion_step):
        im =  Image.fromarray((torch.permute(sample_image[i][j],(1,2,0))*255.0).numpy().astype(np.uint8))
        ims.append(im.quantize())

    # import ipdb;ipdb.set_trace()

    save_name = img_save_path+f"/sample_{i}_diffusion.gif"
    ims[0].save(save_name, save_all=True, append_images=ims[1:], optimize=False, duration=50, loop=0)
    ims[-1].save(img_save_path+f"/sample_{i}.png")



    # aa = to_torch(np.asarray(pil_image_))
    # bb = to_torch(batch_images[i][-1])


    target_image_mini_batch = box_array_handler.get_2d_image_to_mini_batch_image(np.asarray(pil_image_),"x")
    sample_image_mini_batch = box_array_handler.get_2d_image_to_mini_batch_image(batch_images[i][-1],"x")

    mse_loss_fn = nn.MSELoss()
    total_loss = []
    for k in range(target_image_mini_batch.shape[0]):
        if k in slice_tag:
            h=1
        else:
            aa = to_torch(target_image_mini_batch[k])
            bb = to_torch(sample_image_mini_batch[k])
            loss = to_np(mse_loss_fn(aa,bb))
            total_loss.append(loss)
            # print(f"idx:{k} | loss:{loss}")
    loss = np.asarray(total_loss).mean()
    print(f"loss:{loss}")


    # import ipdb;ipdb.set_trace()
    # loss = nn.MSELoss()
    # loss =to_np( loss(aa,bb))

    data = {"denoising_process" : batch_images[i],
            "loss"              : loss,
            "slice_tag"         : slice_tag,
            "s_grid_config"     : s_grid_config,
            "gt_mesh"           : merged}
    pickle_utils().save(data,save_path=img_save_path+f"/denoising_process_{i}.pickle")
    del ims


hoge = (torch.permute(sample_image[0][-2],(1,2,0))).numpy()
hoge = hoge.clip(0,1,hoge)
# hoge = np.random.rand(64,64,3)
updated_colors = box_array_handler.cast_2d_image_to_box_color(image=hoge,permute="z")
colors = updated_colors

import ipdb;ipdb.set_trace()