

import os

import torch
from torch import nn



import numpy as np
import os
from PIL import Image,ImageDraw
import pyvista as pv

from denoising_diffusion_pytorch.utils.config import Config

from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env

from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.serialization import load_diffusion


from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy



# from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array
from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array_multi_type_obj
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml



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

slice_tag         = [5]



print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"



def get_action_candidates(grid_config):

    image_length = grid_config["side_length"]
    aa  = {}
    h   = 0

    data_order = ["x","y","z"]
    for val in range(len(data_order)):
        for j in range(image_length):
            aa.update({h:{"axis":data_order[val],"loc":j}})
            h+=1

    return aa



if __name__ == '__main__':


    #---------------------------------- setup ----------------------------------#


    class Parser(parser):
        dataset: str = 'Image_diffusion_2D'
        config: str =  'config.vae'


    args = Parser().parse_args('diffusion_plan')


    #---------------------------------- loading ----------------------------------#
    ## diffusion model load
    diffusion_experiment    = load_diffusion(args.diffusion_loadpath, epoch=args.diffusion_epoch)
    diffusion               = diffusion_experiment.ema
    dataset                 = diffusion_experiment.dataset


    #--------------------------------- ----------------------------------#


    sample_image_num            = args.batch_size
    test_save_folder            = args.savepath
    cond_save_path              = os.path.normpath(f"{test_save_folder}/voxel_images")
    create_folder(cond_save_path)





    #---------------------------load evaluation model ---------------------------#
    dataset_path                =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/Boxy_99"
    mesh_config_path            =  f"{dataset_path}/generated_configs_w_multi_color.yaml"
    mesh_config                 = load_yaml(mesh_config_path)


    mesh_components = {}
    for  idx, val in enumerate(mesh_config["inner_box"]):
        if  "Component" in val:
            mesh_path   = f"{dataset_path}/blend/Boxy_0_cut0_{val}.stl"
            mesh        = pv.read(mesh_path)
            data        = { val:{"mesh" :mesh,
                                "color" :mesh_config["inner_box"][val]['color']}
                            }
            print(f"load: mesh_path | {mesh_path}, color | {data[val]['color']}")
            mesh_components.update(data)
        else:
            pass







    ## voxelize mesh and get sliced image
    s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
                        "side_length":16}


    oracle_obs_model = voxel_cut_handler(grid_config=s_grid_config, mesh_components=mesh_components,zero_initialize=False)
    seq_obs_model    = voxel_cut_handler(grid_config=s_grid_config, mesh_components=mesh_components,zero_initialize=True)
    policy_obs_model = voxel_cut_handler(grid_config=s_grid_config, mesh_components=mesh_components,zero_initialize=True)


    policy_ = Config(
        args.policy,
        diffusion   = diffusion,
        sample_image_num= args.batch_size,
        obs_model   =policy_obs_model,
        savepath    = (args.savepath, 'policy_config.pkl'),
    )

    policy = policy_()


    action_table = get_action_candidates(grid_config=s_grid_config)

    observation_history = {}


    env = dismantling_env(grid_config=s_grid_config,mesh_components=mesh_components)
    obs,reward,done,info = env.reset()


    pil_image_save_from_numpy(oracle_obs_model.init_imgs_z,f"{cond_save_path}/oracle_obs_cast_z_axis{0}.png")
    pil_image_save_from_numpy(oracle_obs_model.init_imgs_x,f"{cond_save_path}/oracle_obs_cast_x_axis{0}.png")
    pil_image_save_from_numpy(oracle_obs_model.init_imgs_y,f"{cond_save_path}/oracle_obs_cast_y_axis{0}.png")

    pil_image_save_from_numpy(seq_obs_model.init_imgs_z,f"{cond_save_path}/seq_obs_cast_z_axis{0}.png")
    pil_image_save_from_numpy(seq_obs_model.init_imgs_x,f"{cond_save_path}/seq_obs_cast_x_axis{0}.png")
    pil_image_save_from_numpy(seq_obs_model.init_imgs_y,f"{cond_save_path}/seq_obs_cast_y_axis{0}.png")



    # action_1 =  { "axis": "x", "loc" : 1}
    action_idx          = 1
    action              =  action_table[action_idx]
    mini_batch_image_x  = oracle_obs_model.get_obs(action= action)
    seq_obs_model.update_color(mini_batch_image=mini_batch_image_x,config=action)
    observation_history.update({action_idx:action})



    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="z"),f"{cond_save_path}/seq_obs_cast_z_axis{1}.png")
    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="x"),f"{cond_save_path}/seq_obs_cast_x_axis{1}.png")
    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="y"),f"{cond_save_path}/seq_obs_cast_y_axis{1}.png")




    # action_2    =  { "axis": "x", "loc" : 6}
    action_idx  = 6
    action      =  action_table[action_idx]
    obs_image   = oracle_obs_model.get_obs(action= action)
    seq_obs_model.update_color(mini_batch_image=obs_image,config=action)
    observation_history.update({action_idx:action})


    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="z"),f"{cond_save_path}/seq_obs_cast_z_axis{2}.png")
    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="x"),f"{cond_save_path}/seq_obs_cast_x_axis{2}.png")
    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="y"),f"{cond_save_path}/seq_obs_cast_y_axis{2}.png")


    # action_3    =  { "axis": "z", "loc" : 3}
    action_idx  = 35
    action      =  action_table[action_idx]
    obs_image   = oracle_obs_model.get_obs(action= action)
    seq_obs_model.update_color(mini_batch_image=obs_image,config=action)
    observation_history.update({action_idx:action})

    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="z"),f"{cond_save_path}/seq_obs_cast_z_axis{3}.png")
    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="x"),f"{cond_save_path}/seq_obs_cast_x_axis{3}.png")
    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="y"),f"{cond_save_path}/seq_obs_cast_y_axis{3}.png")




    slice_img = seq_obs_model.get_2d_image(axis="z")
    next_action, sorted_action = policy.get_optimal_act(slice_img,observation_history)


    action_idx  = next_action
    action      =  action_table[action_idx]
    obs_image   = oracle_obs_model.get_obs(action= action)
    seq_obs_model.update_color(mini_batch_image=obs_image,config=action)
    observation_history.update({action_idx:action})

    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="z"),f"{cond_save_path}/seq_obs_cast_z_axis{4}.png")
    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="x"),f"{cond_save_path}/seq_obs_cast_x_axis{4}.png")
    pil_image_save_from_numpy(seq_obs_model.get_2d_image(axis="y"),f"{cond_save_path}/seq_obs_cast_y_axis{4}.png")



    slice_img = seq_obs_model.get_2d_image(axis="z")
    next_action, sorted_action = policy.get_optimal_act(slice_img,observation_history)


    import ipdb;ipdb.set_trace()



    # normalizer      = LimitsNormalizer(slice_img)
    # normalized_cond = normalizer.normalize(slice_img).transpose(2,0,1)
    # normalized_cond = to_torch(normalized_cond)

    # cond ={0:{ "idx":torch.where(normalized_cond>-1.0),
    #            "val":normalized_cond,
    #            "obs":action_1}}


    # sample_image = diffusion.model.sample(batch_size=sample_image_num,return_all_timesteps=True, cond = cond ).detach().cpu()
    # batch_size, diffusion_step,image_dim,image_dim,channels = sample_image.shape
    
    
    # # make denoising process gif
    # batch_images = (torch.permute(sample_image,(0,1,3,4,2))*255.0).numpy().astype(np.uint8)
    # last_step_images            = batch_images[:,-1,:,:,:]
    # last_step_images_ensemble   = last_step_images.mean(0)/255.0
    # pil_image_save_from_numpy(last_step_images_ensemble,"./hoges.png")



    # import ipdb;ipdb.set_trace()



    # for i in range(batch_size):
    #     img_save_path = os.path.normpath(f"{test_save_folder}/batch_{i}")
    #     create_folder(img_save_path)

    #     ims = []
    #     for j in range(diffusion_step):
    #         im =  Image.fromarray((torch.permute(sample_image[i][j],(1,2,0))*255.0).numpy().astype(np.uint8))
    #         ims.append(im.quantize())

    #     # import ipdb;ipdb.set_trace()

    #     save_name = img_save_path+f"/sample_{i}_diffusion.gif"
    #     ims[0].save(save_name, save_all=True, append_images=ims[1:], optimize=False, duration=50, loop=0)
    #     ims[-1].save(img_save_path+f"/sample_{i}.png")



    #     # aa = to_torch(np.asarray(pil_image_))
    #     # bb = to_torch(batch_images[i][-1])


    #     target_image_mini_batch = box_array_handler.get_2d_image_to_mini_batch_image(np.asarray(pil_image_),"x")
    #     sample_image_mini_batch = box_array_handler.get_2d_image_to_mini_batch_image(batch_images[i][-1],"x")

    #     mse_loss_fn = nn.MSELoss()
    #     total_loss = []
    #     for k in range(target_image_mini_batch.shape[0]):
    #         if k in slice_tag:
    #             h=1
    #         else:
    #             aa = to_torch(target_image_mini_batch[k])
    #             bb = to_torch(sample_image_mini_batch[k])
    #             loss = to_np(mse_loss_fn(aa,bb))
    #             total_loss.append(loss)
    #             # print(f"idx:{k} | loss:{loss}")
    #     loss = np.asarray(total_loss).mean()
    #     print(f"loss:{loss}")


    #     # import ipdb;ipdb.set_trace()
    #     # loss = nn.MSELoss()
    #     # loss =to_np( loss(aa,bb))

    #     data = {"denoising_process" : batch_images[i],
    #             "loss"              : loss,
    #             "slice_tag"         : slice_tag,
    #             "s_grid_config"     : s_grid_config,
    #             "gt_mesh"           : merged}
    #     pickle_utils().save(data,save_path=img_save_path+f"/denoising_process_{i}.pickle")
    #     del ims


    # hoge = (torch.permute(sample_image[0][-2],(1,2,0))).numpy()
    # hoge = hoge.clip(0,1,hoge)
    # # hoge = np.random.rand(64,64,3)
    # updated_colors = box_array_handler.cast_2d_image_to_box_color(image=hoge,permute="z")
    # colors = updated_colors

    import ipdb;ipdb.set_trace()