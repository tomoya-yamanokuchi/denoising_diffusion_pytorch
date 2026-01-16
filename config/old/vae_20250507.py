

import numpy as np
from denoising_diffusion_pytorch.utils import watch




#------------------------ base ------------------------#
diffusion_train_args_to_watch = [
    ('prefix', ''),
    ('n_diffusion_step',  'T'),
    ('sampling_timestep', 'S'),
    ('image_size', 'D'),
    ('tag', ''),
]



diffusion_1d_train_args_to_watch = [
    ('prefix', ''),
    ('n_diffusion_step',  'T'),
    ('sampling_timestep', 'S'),
    ('image_size', 'D'),
    ('tag', ''),
]


diffusion_plan_args_to_watch = [
    ('prefix', ''),
    ('diffusion_epoch', 'PT'),
    # ('batch_size', 'B'),
    ('tag', ''),
]


cvae_train_args_to_watch = [
    ('prefix', ''),
    ('image_size', 'D'),
    ('latent_dim', 'L'),
    ('tag', ''),
]



vaeac_train_args_to_watch = [
    ('prefix', ''),
    ('image_size', 'D'),
    # ('latent_dim', 'L'),
    ('tag', ''),
]


base = {

    'grid_config' : { 's_grid_config' : {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05), 'side_length':16}
        },


    'diffusion': {
        # 'USER_NAME'         : "haxhi",
        'USER_NAME'         : "user",
        
        ## model
        'model'             : 'models.unet_2d.Unet',
        'dim_mults'         : (1, 2, 4, 8),
        # 'dim_mults'         : (1, 2, 2, 4, 8),
        'flash_attn'        : True,
        'self_condition'    : False,
        # 'self_condition'    : True,

        'diffusion'         : 'models.diffusion.GaussianDiffusion',
        'beta_schedule'     : 'sigmoid', # default
        # 'beta_schedule'     : 'cosine',
        'n_diffusion_step'  : 1000,
        # 'sampling_step'     : 500,
        'sampling_step'     : 20,


        ## dataset
        'loader'            : "data_loader.image_data_loader.Dataset",
        # 'image_size'        : 64 , # for 24GB GPU
        # 'image_size'        : 128 , # for 24GB GPU
        # 'image_size'        :  256,
        'image_size'        :  344,
        # 'image_size'        :  512,
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_3_12900k/voxel_images_w_multi_color_v1',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/voxel_images_w_multi_color_v2',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/voxel_images_w_multi_color_v1',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_45_12900k/voxel_images_w_multi_color_v1',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_41424351_12900k/voxel_images_w_multi_color_v1',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4142435161_13900k/voxel_images_w_multi_color_v1',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4142435161_13901k/voxel_images_w_multi_color_v1',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_41435161_13900k/voxel_images_w_multi_color_v1',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/SheetSander/random_generation_v1/cast_images',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v1',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v1/cast_images_polisher',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v1/cast_images',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v1/cast_images_kw_256',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v1/cast_images_49',
        'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v2/sheetsander_cast_images_z_49',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/flower-image/train_concat',
        # 'horizontal_flip'   : True,
        'horizontal_flip'   : False,
        'convert_image_to'  : "RGB",

        ## serialization
        'logbase'           : 'logs',
        'prefix'            : 'diffusion/',
        # 'prefix'            : 'diffusion_flower/',
        # 'tag'               : "flower_image_v1",
        # 'tag'               : "dataset_2_12900k_v3",
        # 'tag'               : "dataset_4_12900k_v2",
        # 'tag'               : "dataset_45_12900k_v1",
        # 'tag'               : "dataset_41424351_12900k_v4",
        # 'tag'               : "dataset_4142435161_13900k_v7",
        # 'tag'               : "dataset_4142435161_13900k_v8",
        # 'tag'               : "dataset_4142435161_13901k_v1",
        # 'tag'               : "dataset_41435161_13900k_v1",
        # 'tag'               : "real_models_dataset_v2",
        # 'tag'               : "H100_real_models_dataset_v1_4",
        # 'tag'               : "H100_real_models_dataset_v1_5",
        # 'tag'               : "H100_real_models_dataset_v1_6",
        # 'tag'               : "H100_real_models_dataset_v1_7",
        'tag'               : "H100_real_models_dataset_v2_2",
        # 'tag'               : "temp_v1",

        'exp_name'          : watch(diffusion_train_args_to_watch),

        ## training
        'trainer'           : 'trainer.diffusion_trainer.Trainer',
        # 'batch_size'        : 64, # default
        # 'batch_size'        : 32,
        'batch_size'        : 18,
        # 'batch_size'        : 8,
        'learning_rate'     : 8e-5,
        # 'learning_rate'     : 9e-5,
        'train_step'        : 800000,           # total training steps
        # 'gradient_accumulate_every' : 2,        # gradient accumulation steps
        # 'gradient_accumulate_every' : 3,        # gradient accumulation steps
        'gradient_accumulate_every' : 5,        # gradient accumulation steps
        # 'gradient_accumulate_every' : 6,        # gradient accumulation steps
        'ema_decay'          : 0.995,           # exponential moving average decay
        'amp'                : True,            # turn on mixed precision
        'calculate_fid'      : False,
        'device'             :'cuda:0'
    },

    'diffusion_1d': {
        'USER_NAME'         : "haxhi",
        ## model
        'model'             : 'models.unet_1d.Unet1D',
        # 'dim'               : 64, # default
        'dim'               : 128,
        'dim_mults'         : (1, 2, 2, 4, 8),
        'self_condition'    : False,
        # 'flash_attn'        : True,
        # 'self_condition'    : True,

        'diffusion'         : 'models.diffusion_1d.GaussianDiffusion1D',
        'beta_schedule'     : 'sigmoid', # default
        # 'beta_schedule'     : 'cosine',
        'n_diffusion_step'  : 1000,
        # 'sampling_step'     : 500,
        'sampling_step'     : 20,


        ## dataset
        'loader'            : "data_loader.image_data_loader.Dataset1D",
        'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4142435161_13901k/voxel_images_w_multi_color_v1',
        'image_size'        : 64,
        'grid_3dim'         : 16,
        'is_shuffle'        : True,
        'horizontal_flip'   : False,
        'convert_image_to'  : None,


        ## serialization
        'logbase'           : 'logs',
        'prefix'            : 'diffusion_1d/',
        # 'prefix'            : 'diffusion_flower/',
        # 'tag'               : "flower_image_v1",
        # 'tag'               : "dataset_2_12900k_v3",
        # 'tag'               : "dataset_4_12900k_v2",
        # 'tag'               : "dataset_45_12900k_v1",
        # 'tag'               : "dataset_41424351_12900k_v4",
        # 'tag'               : "dataset_4142435161_13900k_v7",
        # 'tag'               : "dataset_4142435161_13900k_v8",
        'tag'               : "dataset_4142435161_13901k_v2/",
        # 'tag'               : "dataset_41435161_13900k_v1",

        'exp_name'          : watch(diffusion_1d_train_args_to_watch),

        ## training
        'trainer'           : 'trainer.diffusion_1d_trainer.Trainer1D',
        'batch_size'        : 64,
        'learning_rate'     : 8e-5,
        'train_step'        : 800000,           # total training steps
        'gradient_accumulate_every' : 2,        # gradient accumulation steps
        'ema_decay'          : 0.995,           # exponential moving average decay
        'amp'                : True,            # turn on mixed precision
        'device'             :'cuda'
    },


    'cvae': {
        'USER_NAME'         : "haxhi",
        'device'             :'cuda',

        ## model
        'model'             : 'models.cvae.cvae_v1.VAE_2dim_conv',
        'latent_dim'        : 64,

        'cvae'              : 'models.cvae.cvae_handler.VAE_Handler_2dim_conv',


        ## dataset
        'loader'            : "data_loader.cvae_data_loader.cvaedataset",
        'image_size'        : 64 ,
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_3_12900k/voxel_images_w_multi_color_v1',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/voxel_images_w_multi_color_v2',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/voxel_images_w_multi_color_v1',
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/flower-image/train_concat',
        'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/img_align_celeba/',

        'horizontal_flip'   : True,
        'convert_image_to'  : "RGB",

        ## serialization
        'logbase'           : 'logs',
        'prefix'            : 'cvae/',
        # 'prefix'            : 'diffusion_flower/',
        # 'tag'               : "flower_image_v1",
        # 'tag'               : "dataset_2_12900k_v3",
        # 'tag'               : "dataset_4_12900k_v2",
        'tag'               : "image_align_celeba_2",
        'exp_name'          : watch(cvae_train_args_to_watch),

        ## training
        'trainer'           : 'trainer.cvae_trainer.Trainer',

        'batch_size'         : 16,
        'learning_rate'      : 2e-5,
        'log_freq'           : 100,
        'sample_freq'        : 10000,
        'save_freq'          : 10000,
        'label_freq'         : 1000,
        'n_samples'          : 16,
        'n_epoch'            : 1000,
    },


    'vaeac' : {

        # 'USER_NAME'         : "haxhi",
        'USER_NAME'         : "user",
        'device'             :'cuda:0',

        ## dataset
        "loader":"data_loader.vaeac_data_loader.VAEAC_dataloader",
        "dataset_config": {"dataset" : {"name": "celeba",
                                        # "path": "/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_4142435161_13901k/voxel_images_w_multi_color_v1",
                                        "path": '/home/user/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v2/sheetsander_cast_images_z_49',
                                        "min": -1,
                                        "max": 1,
                                        "h": 32,
                                        "type": "pattern",
                                        "p": 0.2}},
        # 'image_size' : 64,
        'image_size' : 344,
        

        ### model
        "model" : 'models.vaeac.vaeac.EncoderDecoder',
        "model_config": {"model": { "name": "EncoderDecoderNet",
                                    "last_layer": "tanh",
                                    "inp_channels": 3,
                                    # "n_hidden": 32,
                                    "n_hidden": 96,
                                    # "fc_hidden": 100,
                                    "fc_hidden": 300,
                                    # "fc_out": 50,
                                    "fc_out": 150,
                                    "optimizer": "adam",
                                    # "lr": 0.0001,
                                    "lr": 0.00005,
                                    # "lr": 8e-5,
                                    "beta1": 0.9,
                                    "beta2": 0.999,
                                    "scheduler": "step",
                                    "decay-steps": 500,
                                    "decay-factor": 0.995,
                                    "weight-decay": 0.00001,
                                    "init": "orthogonal",
                                    "loss": "mse",
                                    "lr_override": False},
                        "reg": {"lambda_kl": 1,
                                "lambda_reg": 1,
                                "sigma_m": 10000,
                                "sigma_s": 0.0001,
                                "apply_reg": True}
                        },
        ## trainer
        'trainer' : 'trainer.vaeac_trainer.Trainer',
        'train_config':{
                        "pretrained": None,
                        "train": {
                            "train_step": 800000,
                            # "save-freq": 2000,
                            # "batch_size": 64,
                            "batch_size": 18,
                            'gradient_accumulate_every' : 5,        # gradient accumulation steps
                            "shuffle": True,
                            "step-log": 5,
                            'peek-validation': 2000}
                        },

        ## serialization
        'logbase'           : 'logs',
        'prefix'            : 'vaeac/',
        # 'tag'               : "image_align_celeba_s2",
        'tag'               : "H100_real_models_dataset_v2_2",
        'exp_name'          : watch(vaeac_train_args_to_watch),
    },




    'diffusion_plan': {
        # 'USER_NAME'         : "haxhi",
        'USER_NAME'         : "user",

        'policy'            :'policy.cutting_surface_planner_v7.cutting_surface_planner',
        # 'batch_size'        : 16,
        # 'batch_size'        : 32, # default
        'batch_size'        : 32,
        # 'batch_size'        : 64,


        ## policy_config
        'policy_config'     :{
                                'ctrl_mode':"epsilon_greedy_00",
                                # 'ctrl_mode':"epsilon_greedy_001",
                                # 'ctrl_mode':"epsilon_greedy_01",
                                # 'ctrl_mode':"epsilon_greedy_05",
                                # 'ctrl_mode':"random",
                                # 'ctrl_mode':"no_cond",
                                # 'ctrl_mode':"oracle_obs",
                                "image_mask_config_b" : {"target_mask"  :np.asarray([0.2,0.8,0.8]),
                                                        "target_mask_lb":np.asarray([0.2,0.8,0.8])-np.asarray([0.1,0.1,0.1]),
                                                        "target_mask_ub":np.asarray([0.2,0.8,0.8])+np.asarray([0.2,0.2,0.2])}, #np.asarray([0.7,0.2,0.2]
                                "image_mask_config_r": {"target_mask"   :np.asarray([0.8,0.2,0.2]),
                                                        "target_mask_lb":np.asarray([0.8,0.2,0.2])-np.asarray([0.1,0.1,0.1]),
                                                        "target_mask_ub":np.asarray([0.8,0.2,0.2])+np.asarray([0.2,0.2,0.2])}, #np.asarray([0.2,0.6,0.6])
                                "image_mask_config_y": {"target_mask"   :np.asarray([0.8,0.8,0.2]),
                                                        "target_mask_lb":np.asarray([0.8,0.8,0.2])-np.asarray([0.1,0.1,0.1]),
                                                        "target_mask_ub":np.asarray([0.8,0.8,0.2])+np.asarray([0.2,0.2,0.6])},
                                # 'infer_model':"vaeac",
                                'infer_model':"diffusion",
                                # 'infer_model':"diffusion_1D",
                                # 'decision_mode':"cal_cost_mean",
                                # 'decision_mode':"cal_cost_mode",
                                # 'decision_mode':"remove_outliers_cal_cost_mean",
                                "decision_mode": {  "mode": "cal_cost_mean_ucb",
                                                    "param":{"cost_lb_discount_factor":0.99}},
                                },
        ##eval data loading
        # 'eval_data_path'    : 'f:/home/{USER_NAME}/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/',
        # 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/nedo_dismantling_dataset/dataset_1/geom_eval/',
        # 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_1/geom_eval',
        # 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_2_12900k/geom_test',
        # 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test',
        # 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_2',
        'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_3',
        # 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1',
        # 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1',





        ## loading
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising-diffusion-pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_v0',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_test_2',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_2_12900k_v1',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_2_12900k_v2',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_3_12900k_v2',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_4_12900k_v1',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_4_12900k_v2',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_5_12900k_v1',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_45_12900k_v1',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_41424351_12900k_v1',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_41424351_12900k_v4',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_4142435161_13900k_v1',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_4142435161_13900k_v2',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_4142435161_13900k_v2',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_4142435161_13900k_v5',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_4142435161_13900k_v7',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_4142435161_13900k_v8',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_4142435161_13901k_v1',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_41435161_13900k_v1',
        # 'diffusion_epoch'   : '40000',
        # 'diffusion_epoch'   : '80000',
        # 'diffusion_epoch'   : '600000',
        # 'diffusion_epoch'   : '100',
        # 'diffusion_epoch'   : '140000',
        # 'diffusion_epoch'   : '190000',
        # 'diffusion_epoch'   : '200000',
        # 'diffusion_epoch'   : '400000',
        # 'diffusion_epoch'   : '600000',
        # 'diffusion_epoch'   : '800000',


        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_dataset_4142435161_13901k_v1',
        # 'diffusion_epoch'   : '200000',
        # 'prefix'            : 'diffusion_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/',

        'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D256_H100_real_models_dataset_v1_3',
        'diffusion_epoch'   : '102000',
        'prefix'            : 'diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/',

        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion/T1000_D512_H100_real_models_dataset_v1_1',
        # 'diffusion_epoch'   : '81000',
        # # 'diffusion_epoch'   : '50000',
        # 'prefix'            : 'diffusion_plans/real_model/real_models_dataset_v1_1/dataset_4_3_eval/',
        


        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_1d/T1000_D64_dataset_4142435161_13901k_v1/',
        # 'diffusion_epoch'   : '200000',
        # 'prefix'            : 'diffusion_plans_1d/dataset_4142435161_13901k_v1/dataset_4_3_eval/',


        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac/D64_image_align_celeba_s2',
        # 'diffusion_epoch'   : '200000',
        # 'prefix'            : 'vaeac_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/',



        # 'task_step' :10,
        'task_step' :15,
        'start_action_idx':19,
        # 'start_action_idx':41, # for dataset_eval_4_3 -> action index = 41
        # 'start_action_idx':4,


        ## serialization
        'logbase'           : 'logs',
        'suffix'            : 'f:{policy_config["ctrl_mode"]}',
        'iter'              : [0,6],
        # 'iter'              : [0,2],
        # 'iter'              : [0,10],
        'tag'               : 'f:T1000_D64_B{batch_size}_test_v6_fix_start_multi_step_partial_obs_a{start_action_idx}_long_{policy_config["infer_model"]}_{policy_config["decision_mode"]["mode"]}_9_t20_8_3_tmp12_1',
        'exp_name'          : watch(diffusion_plan_args_to_watch),
        'device': 'cuda:0',
        
        ### tmp_8_1: Repaint version conditioning
        ### tmp 8_2: hard conditioning 
        ### tmp 8_3: Repaint version conditioning with disable clamp
        ### tmp_8_4: tmp 8_3 setting with transforms.Resize((343, 343), interpolation=InterpolationMode.NEAREST), 
        ### tmp_8_5: tmp 8_2 setting with transforms.Resize((343, 343), interpolation=InterpolationMode.NEAREST), 
        
        ### tmp_9_4: Repaint version condition disabe clamp with transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST), 
        ### tmp_9_5: hard_conditioning with transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST), 
        ### tmp_9_6: 9_4 with sampling = 50, 
        ### tmp_9_7: 9_5 with sampling = 50, 
        
        ### tmp_10_1 : tmp8_3 with 512 512 raw model
        ### tmp_10_2 : tmp8_2 with 512 512 raw model
        
        # dataset_v1_3
        ## tmp_11_1 : tmp_8_3 with 256, transforms.Resize((model_img_train_dim,model_img_train_dim),interpolation=InterpolationMode.NEAREST), transforms.Resize((env_img_dim, env_img_dim), interpolation=InterpolationMode.NEAREST),  last 5 step cond is none
        ## tmp_11_2 : tmp_8_3 with 256, transforms.Resize((model_img_train_dim,model_img_train_dim),), transforms.Resize((env_img_dim, env_img_dim), interpolation=InterpolationMode.NEAREST),  last one step cond is none
        ## tmp_11_3 : tmp_8_3 with 256, transforms.Resize((model_img_train_dim,model_img_train_dim),), transforms.Resize((env_img_dim, env_img_dim), interpolation=InterpolationMode.NEAREST),  last 5 step cond is none
        
        
        
        
    },
}



Reacher_v2 = {
}