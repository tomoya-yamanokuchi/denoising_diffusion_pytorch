

import os
from tqdm import tqdm
from multiprocessing import cpu_count

import torch
from torch import nn

import random

import numpy as np
import os
from PIL import Image,ImageDraw
import pyvista as pv

from denoising_diffusion_pytorch.utils.config import Config

from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env

from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.serialization import load_diffusion,load_vaeac


from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy



# from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml





print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_eval_data(data_source_dir):

    eval_folders    = get_folder_name(data_source_dir)[:3]
    eval_data       = {}
    for i in range(len(eval_folders)):
        dataset_path                =  f"{data_source_dir}"+f"/{eval_folders[i]}"
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
        eval_data.update({f"{eval_folders[i]}":mesh_components})

    return eval_data


if __name__ == '__main__':


    #---------------------------------- setup ----------------------------------#


    class Parser(parser):
        dataset: str = 'Image_diffusion_2D'
        config: str  = 'config.vae'


    args = Parser().parse_args('diffusion_plan')
    grid_config = Parser().parse_args('grid_config')

    original_config_path = args.savepath
    original_config_path = os.path.join(original_config_path,"configs_backup.py")
    args.save_config_file(original_config_path)

    #---------------------------------- loading ----------------------------------#

    # import ipdb;ipdb.set_trace()

    ## diffusion model load
    if  args.policy_config["infer_model"] == 'vaeac':
        diffusion_experiment    = load_vaeac(args.diffusion_loadpath, epoch=args.diffusion_epoch)
    elif args.policy_config["infer_model"] == 'diffusion' or args.policy_config["infer_model"] == 'diffusion_1D' or args.policy_config["infer_model"] == 'conditional_diffusion':
        diffusion_experiment    = load_diffusion(args.diffusion_loadpath, epoch=args.diffusion_epoch)
    diffusion               = diffusion_experiment.ema
    dataset                 = diffusion_experiment.dataset
    trainer                 = diffusion_experiment.trainer


    #--------------------------------- ----------------------------------#
    sample_image_num            = args.batch_size
    test_save_folder            = args.savepath

    #---------------------------load evaluation model ---------------------------#
    eval_data_dir               =  args.eval_data_path
    eval_dataset                =  get_eval_data(eval_data_dir)



    ## voxe setting
    s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
                        "side_length":16}
    
    # s_grid_config = grid_config.s_grid_config



    for idx, val in enumerate(eval_dataset):

        mesh_components  = eval_dataset[val]
        policy_obs_model = voxel_cut_handler(grid_config=s_grid_config, mesh_components=mesh_components,zero_initialize=True)



        for episode_num in tqdm(range(args.iter[0],args.iter[1])):
        # for episode_num in tqdm(range(1)):


            policy_ = Config(
                args.policy,
                verbose  = False,
                diffusion   = diffusion,
                trainer     = trainer,
                sample_image_num= args.batch_size,
                obs_model   = policy_obs_model,
                config      = args.policy_config,
                # savepath    = (args.savepath, 'policy_config.pkl'),
            )
            policy = policy_()



            cond_save_path = os.path.normpath(f"{test_save_folder}/{val}/episode_{episode_num}")
            # cond_save_path = os.path.normpath(f"{test_save_folder}/random/episode_{episode_num}")
            create_folder(cond_save_path)

            # import ipdb;ipdb.set_trace()

            ## env for evaluation
            env = dismantling_env(grid_config=s_grid_config,mesh_components=mesh_components)
            obs,reward,done,info = env.reset()

            ## env for policy partial observation
            env2 = dismantling_env(grid_config=s_grid_config,mesh_components=mesh_components)
            obs,reward,done,info = env2.reset()

            pil_image_save_from_numpy(info["oracle_obs"]["x"],f"{cond_save_path}/oracle_obs_cast_x_axis{0}.png")
            pil_image_save_from_numpy(info["oracle_obs"]["y"],f"{cond_save_path}/oracle_obs_cast_y_axis{0}.png")
            pil_image_save_from_numpy(info["oracle_obs"]["z"],f"{cond_save_path}/oracle_obs_cast_z_axis{0}.png")

            action_l    = []
            reward_l    = []
            obs_l       = []
            info_l      = []
            removal_pref_l =  []



            for i in range(int(args.task_step)):
                if i == 0:
                    cut_cost_tmp    = 0
                    action_         = args.start_action_idx
                    # action_ = random_number = random.randint(0, 47)
                    obs, reward, done, info = env.step(action_idx=action_)
                    env2.step(action_idx=action_)
                    print(f"init_action:{action_}")
                else:
                    print(f"action_slice_range:{action}")
                    cut_cost_tmp = 0
                    for j in range(len(action)):
                        action_ = action[j]
                        obs,reward,done,info = env.step(action_idx=action_)
                        print(f'step: {j} | cut_cost: {reward} | target_removal_rate {info["target_removal_rate"]}| removal performance :{info["removal_performance"]:.3f}')
                        cut_cost_tmp+=reward


                reward = cut_cost_tmp
                action_l.append(action_)
                reward_l.append(reward)
                obs_l.append(obs["sequential_obs"]["z"])
                info_l.append(info["target_removal_rate"])
                removal_pref_l.append(info["removal_performance"])

                # print(f'step: {i} | cut_cost: {reward} | target_removal_rate {info["target_removal_rate"]}| obs_history : {obs["observation_history"]}')
                print(f'step: {i} | cut_cost: {reward} | target_removal_rate {info["target_removal_rate"]}| removal performance :{info["removal_performance"]:.3f}')
                # print(f'step: {i} | obs_history : {obs["observation_history"]}')

                pil_image_save_from_numpy(obs["sequential_obs"]["x"],f"{cond_save_path}/{i}_seq_obs_cast_x_axis{i}_{0}.png")
                pil_image_save_from_numpy(obs["sequential_obs"]["y"],f"{cond_save_path}/{i}_seq_obs_cast_y_axis{i}_{0}.png")
                pil_image_save_from_numpy(obs["sequential_obs"]["z"],f"{cond_save_path}/{i}_seq_obs_cast_z_axis{i}_{0}.png")

                # import ipdb;ipdb.set_trace()

                policy.set_oracle_obs(info["oracle_obs"]["z"])
                # next_action, sorted_action, infos = policy.get_optimal_act(obs["sequential_obs"]["z"],obs["observation_history"])

                if i == 0:
                    next_action, sorted_action, infos = policy.get_optimal_act(obs["sequential_obs"]["z"],obs["observation_history"],env2,action_,i,cond_save_path)
                else:
                    next_action, sorted_action, infos = policy.get_optimal_act(obs["sequential_obs"]["z"],obs["observation_history"],env2,action[-1],i,cond_save_path)

                action = next_action

                pil_image_save_from_numpy(infos["ensemble_image"]["z"],f"{cond_save_path}/{i}_ensemble_z_axis{i}_{0}.png")
                pil_image_save_from_numpy(infos["ensemble_image"]["x"],f"{cond_save_path}/{i}_ensemble_x_axis{i}_{0}.png")
                pil_image_save_from_numpy(infos["ensemble_image"]["y"],f"{cond_save_path}/{i}_ensemble_y_axis{i}_{0}.png")


            # import ipdb;ipdb.set_trace()
            action_np = np.asarray(action_l)
            reward_np = np.asarray(reward_l)
            obs_np    = np.asarray(obs_l)
            info_np   = np.asarray(info_l)

            rollout_data={ "observations"           : np.asarray(obs_np),
                            'actions'               : np.asarray(action_l),
                            'rewards'               : np.asarray(reward_l),
                            'infos'                 : np.asarray(info_l),
                            'removal_performance'   : np.asarray(removal_pref_l),
                        }

            pickle_utils().save(dataset=rollout_data,save_path=f"{cond_save_path}/rollout_data.pickle")

        # del policy


    # import ipdb;ipdb.set_trace()

