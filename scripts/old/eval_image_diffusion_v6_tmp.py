

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

from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.benchmark_model_utils import get_benchmark_model

from denoising_diffusion_pytorch.utils.serialization import load_diffusion,load_vaeac


from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy



# from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml





print(torch.__version__)
print(torch.cuda.is_available())
# device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()
torch.cuda.ipc_collect()


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


def get_model(dataset_path, model_config):

        mesh_components = {}

        for idx, val in enumerate(model_config["outer_parts"]):
                print(val)
                mesh_path   =  os.path.normpath(dataset_path+model_config["outer_parts"][val]["path"])
                mesh        = pv.read(mesh_path)
                mesh_color  = model_config["outer_parts"][val]["color"]
                data        = {val:{    "mesh":mesh,
                                        "color":mesh_color}}
                print(f"load: mesh_path | {mesh_path}, color | {data[val]['color']}")
                mesh_components.update(data)


        for idx, val in enumerate(model_config["internal_parts"]):
                print(val)
                mesh_path   =  os.path.normpath(dataset_path+model_config["internal_parts"][val]["path"])
                mesh        = pv.read(mesh_path)
                mesh_color  = model_config["internal_parts"][val]["color"]
                data        = {val:{    "mesh":mesh,
                                        "color":mesh_color}}
                print(f"load: mesh_path | {mesh_path}, color | {data[val]['color']}")
                mesh_components.update(data)

        return mesh_components


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

    ##  model load
    if  args.policy_config["infer_model"] == 'vaeac':
        diffusion_experiment    = load_vaeac(args.diffusion_loadpath, epoch=args.diffusion_epoch)
    elif args.policy_config["infer_model"] == 'diffusion' or args.policy_config["infer_model"] == 'diffusion_1D' or args.policy_config["infer_model"] == 'conditional_diffusion':
        diffusion_experiment    = load_diffusion(args.diffusion_loadpath, epoch=args.diffusion_epoch)
        # diffusion_experiment    = load_diffusion(args.diffusion_loadpath, epoch=args.diffusion_epoch , device="cpu")
        
    diffusion               = diffusion_experiment.ema
    dataset                 = diffusion_experiment.dataset
    trainer                 = diffusion_experiment.trainer


    #--------------------------------- ----------------------------------#
    sample_image_num            = args.batch_size
    test_save_folder            = args.savepath

    #---------------------------load evaluation model ---------------------------#
    eval_data_dir               =  args.eval_data_path
    # eval_dataset                =  get_eval_data(eval_data_dir)



    ## voxe setting
    # s_grid_config = grid_config.s_grid_config
    # s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),"side_length":16}
    s_grid_config = {"bounds":(-0.3,0.3,-0.3,0.3,-0.3,0.3), 'side_length':49}
    # s_grid_config = {"bounds":(-0.3,0.3,-0.3,0.3,-0.3,0.3), 'side_length':64}


    # model_config = {"model_name": "original_Bosch_GPO14CE_Polisher",
    #                 "outer_parts": {
    #                                 "original_Bosch_GPO14CE_PolisherBody1":
    #                                 {"path":"./models/original_Bosch_GPO14CE_PolisherBody1.stl",
    #                                 "color":[0.9,0.9,0.9]},
    #                                 "original_Bosch_GPO14CE_PolisherBody2":
    #                                 {"path":"./models/original_Bosch_GPO14CE_PolisherBody2.stl",
    #                                 "color":[0.9,0.9,0.9]},
    #                                 "original_Bosch_GPO14CE_PolisherBody3":
    #                                 {"path":"./models/original_Bosch_GPO14CE_PolisherBody3.stl",
    #                                 "color":[0.9,0.9,0.9]},
    #                                 "original_Bosch_GPO14CE_PolisherBody4":
    #                                 {"path":"./models/original_Bosch_GPO14CE_PolisherBody4.stl",
    #                                 "color":[0.9,0.9,0.9]},
    #                                 "original_Bosch_GPO14CE_PolisherBody5":
    #                                 {"path":"./models/original_Bosch_GPO14CE_PolisherBody5.stl",
    #                                 "color":[0.9,0.9,0.9]},
    #                                 },
    #                 "internal_parts": {
    #                                 "original_Bosch_GPO14CE_PolisherMotor":
    #                                 {"path":"./models/original_Bosch_GPO14CE_PolisherMotor.stl",
    #                                 "color":[0.9,0.2,0.2]},
    #                                 "original_Bosch_GPO14CE_PolisherPCB":
    #                                 {"path":"./models/original_Bosch_GPO14CE_PolisherPCB.stl",
    #                                 "color":[0.8,0.8,0.2]},
    #                                 "original_Bosch_GPO14CE_PolisherBattery":
    #                                 {"path":"./models/original_Bosch_GPO14CE_PolisherBattery.stl",
    #                                 "color":[0.2,0.8,0.8]},
    #                                 }
    #             }


    # dataset_path   = "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/SheetSander/"
    # with open(os.path.join(dataset_path,"generated_configs.yaml"), encoding='utf-8')as f:
    #     model_config= yaml.safe_load(f)
    # eval_data =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)

    # dataset_path   = "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/PowerCutter/"
    # with open(os.path.join(dataset_path,"generated_configs.yaml"), encoding='utf-8')as f:
    #     model_config= yaml.safe_load(f)
    # eval_data2 =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)

    # dataset_path   = "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/Polisher/"
    # with open(os.path.join(dataset_path,"generated_configs.yaml"), encoding='utf-8')as f:
    #     model_config= yaml.safe_load(f)
    # eval_data3 =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)

    # u_name = "user"
    # dataset_path   = f"/home/{u_name}/dataset/denoising_diffusion_pytorch/dataset/real_models/SheetSander/random_generation_v1/samples/samples_0/"
    # with open(os.path.join(dataset_path,"../../../generated_configs.yaml"), encoding='utf-8')as f:
    #     model_config= yaml.safe_load(f)
    # eval_data =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)

    # dataset_path   = f"/home/{u_name}/dataset/denoising_diffusion_pytorch/dataset/real_models/PowerCutter/random_generation_v1/samples/samples_0/"
    # with open(os.path.join(dataset_path,"../../../generated_configs.yaml"), encoding='utf-8')as f:
    #     model_config= yaml.safe_load(f)
    # eval_data2 =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)

    # dataset_path   = f"/home/{u_name}/dataset/denoising_diffusion_pytorch/dataset/real_models/Polisher/random_generation_v1/samples/samples_0/"
    # with open(os.path.join(dataset_path,"../../../generated_configs.yaml"), encoding='utf-8')as f:
    #     model_config= yaml.safe_load(f)
    # eval_data3 =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)


    u_name = "user"
    dataset_path   = f"/home/{u_name}/dataset/denoising_diffusion_pytorch/dataset/real_models/SheetSander_kwon/random_generation_for_test/samples/samples_0/"
    with open(os.path.join(dataset_path,"SheetSander_type0.yaml"), encoding='utf-8')as f:
        model_config= yaml.safe_load(f)
    eval_data =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)

    dataset_path   = f"/home/{u_name}/dataset/denoising_diffusion_pytorch/dataset/real_models/SheetSander_kwon/random_generation_for_test/samples/samples_2/"
    with open(os.path.join(dataset_path,"SheetSander_type2.yaml"), encoding='utf-8')as f:
        model_config= yaml.safe_load(f)
    eval_data2 =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)

    dataset_path   = f"/home/{u_name}/dataset/denoising_diffusion_pytorch/dataset/real_models/SheetSander_kwon/random_generation_for_test/samples/samples_4/"
    with open(os.path.join(dataset_path,"SheetSander_type4.yaml"), encoding='utf-8')as f:
        model_config= yaml.safe_load(f)
    eval_data3 =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)



    dataset_path   = f"/home/{u_name}/dataset/denoising_diffusion_pytorch/dataset/real_models/SheetSander_kwon/random_generation_for_test/samples/samples_1/"
    with open(os.path.join(dataset_path,"SheetSander_type1.yaml"), encoding='utf-8')as f:
        model_config= yaml.safe_load(f)
    eval_data4 =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)

    dataset_path   = f"/home/{u_name}/dataset/denoising_diffusion_pytorch/dataset/real_models/SheetSander_kwon/random_generation_for_test/samples/samples_3/"
    with open(os.path.join(dataset_path,"SheetSander_type3.yaml"), encoding='utf-8')as f:
        model_config= yaml.safe_load(f)
    eval_data5 =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)

    dataset_path   = f"/home/{u_name}/dataset/denoising_diffusion_pytorch/dataset/real_models/SheetSander_kwon/random_generation_for_test/samples/samples_5/"
    with open(os.path.join(dataset_path,"SheetSander_type5.yaml"), encoding='utf-8')as f:
        model_config= yaml.safe_load(f)
    eval_data6 =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config)


    # eval_dataset = {"Object_1":eval_data,
    #                 "Object_2":eval_data2,
    #                 "Object_3":eval_data3,}
    
    
    # eval_dataset = {
    #                     "Object_3":eval_data3,
    #                     "Object_1":eval_data,
    #                     "Object_2":eval_data2,
    #                     }

    eval_dataset_candidate = {
                    "Object_1":eval_data,
                    "Object_2":eval_data2,
                    "Object_3":eval_data3,
                    "Object_4":eval_data4,
                    "Object_5":eval_data5,
                    "Object_6":eval_data6,}

    start_action_idx= args.start_action_idx

    # 抽出
    eval_dataset = {
        k: eval_dataset_candidate[k] for k in start_action_idx if k in eval_dataset_candidate
    }



    file_path = f"./for_eval_my_dict_{s_grid_config['side_length']}.pkl"

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            load_nearby_cells = pickle.load(f)

    else:
        print("create voxel environments....")
        mesh_components  = eval_dataset["Object_1"]
        tmp = voxel_cut_handler(grid_config=s_grid_config, mesh_components=mesh_components,zero_initialize=False,pre_near_by_cells=None)
        load_nearby_cells = tmp.voxel_hander._create_box_array()
        with open(file_path, 'wb') as f:
            pickle.dump(load_nearby_cells, f)


    for idx, val in enumerate(eval_dataset):

        mesh_components  = eval_dataset[val]
        policy_obs_model = voxel_cut_handler(grid_config=s_grid_config, mesh_components=mesh_components,zero_initialize=True,pre_near_by_cells=load_nearby_cells)



        for episode_num in tqdm(range(args.iter[0],args.iter[1])):
        # for episode_num in tqdm(range(1)):


            policy_ = Config(
                args.policy,
                verbose = False,
                diffusion   = diffusion,
                trainer     = trainer,
                sample_image_num = args.batch_size,
                obs_model   = policy_obs_model,
                observation_mode =  args.observation_mode,
                config      = args.policy_config,
            )
            policy = policy_()




            cond_save_path = os.path.normpath(f"{test_save_folder}/{val}/episode_{episode_num}")
            create_folder(cond_save_path)


            # import ipdb;ipdb.set_trace()

            ## env for evaluation
            env = dismantling_env(grid_config=s_grid_config,mesh_components=mesh_components,pre_near_by_cells=load_nearby_cells)
            obs,reward,done,info = env.reset()

            ## env for policy partial observation
            env2 = dismantling_env(grid_config=s_grid_config,mesh_components=mesh_components,pre_near_by_cells=load_nearby_cells)
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
                    # action_         = args.start_action_idx
                    action_         = args.start_action_idx[val]
                    # action_ = random_number = random.randint(0, 47)
                    if args.observation_mode == "both_cs_obs":
                        env.step(action_idx=action_-1)
                        env2.step(action_idx=action_-1)
                    obs, reward, done, info = env.step(action_idx=action_)
                    env2.step(action_idx=action_)
                    print(f"init_action:{action_}")
                else:
                    print(f"action_slice_range:{action}")
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

                print(f'##########################################################################################################################################')
                print(f'step: {i} | cut_cost: {reward} | target_removal_rate {info["target_removal_rate"]}| removal performance :{info["removal_performance"]:.3f}')
                print(f'##########################################################################################################################################')


                pil_image_save_from_numpy(obs["sequential_obs"]["x"],f"{cond_save_path}/{i}_seq_obs_cast_x_axis{i}_{0}.png")
                pil_image_save_from_numpy(obs["sequential_obs"]["y"],f"{cond_save_path}/{i}_seq_obs_cast_y_axis{i}_{0}.png")
                pil_image_save_from_numpy(obs["sequential_obs"]["z"],f"{cond_save_path}/{i}_seq_obs_cast_z_axis{i}_{0}.png")

                # import ipdb;ipdb.set_trace()

                policy.set_oracle_obs(info["oracle_obs"]["z"])
                # next_action, sorted_action, infos = policy.get_optimal_act(obs["sequential_obs"]["z"],obs["observation_history"])

                if i == 0:
                    if args.observation_mode == "both_cs_obs":
                        policy.get_opposite_site_cutting_surface(env2 = env2, action_idx  = action_-1)
                    next_action, sorted_action, infos = policy.get_optimal_act(obs["sequential_obs"]["z"],obs["observation_history"],env2,action_,i,cond_save_path)
                else:
                    if args.observation_mode == "both_cs_obs":
                        if len(action)>=2:
                            policy.get_opposite_site_cutting_surface(env2 = env2, action_idx  = action[-2])
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

