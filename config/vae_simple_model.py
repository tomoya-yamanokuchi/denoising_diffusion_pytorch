

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


conditional_image_diffusion_train_args_to_watch = [
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
        'dataset_path'      : 'f:/home/dev/workspace/nedo-dismantling-PyBlender/voxel_images_w_multi_color_v2_',
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
        'tag'               : "H100_real_models_dataset_v2_3",
        # 'tag'               : "temp_v1",

        'exp_name'          : watch(diffusion_train_args_to_watch),

        ## training
        'trainer'           : 'trainer.diffusion_trainer.Trainer',
        # 'batch_size'        : 64, # default
        'batch_size'        : 16,
        # 'batch_size'        : 32,
        # 'batch_size'        : 18,
        # 'batch_size'        : 8,
        'learning_rate'     : 8e-5,
        # 'learning_rate'     : 9e-5,
        'train_step'        : 800000,           # total training steps
        'gradient_accumulate_every' : 2,        # gradient accumulation steps
        # 'gradient_accumulate_every' : 3,        # gradient accumulation steps
        # 'gradient_accumulate_every' : 5,        # gradient accumulation steps
        # 'gradient_accumulate_every' : 6,        # gradient accumulation steps
        'ema_decay'          : 0.995,           # exponential moving average decay
        'amp'                : True,            # turn on mixed precision
        'calculate_fid'      : False,
        'device'             :'cuda:0'
    },

    'diffusion_1d': {
        # 'USER_NAME'         : "haxhi",
        'USER_NAME'         : "user",

        ## model
        'model'             : 'models.unet_1d.Unet1D',
        # 'dim'               : 64, # default
        'dim'               : 128,  # simple model setting?
        # 'dim_mults'         : (1, 2, 2, 4, 8), # simple model setting
        'dim_mults'         : (1, 2, 4, 8), # simple model setting
        'self_condition'    : False,
        'flash_attn'        : True,
        # 'self_condition'    : True,

        'diffusion'         : 'models.diffusion_1d.GaussianDiffusion1D',
        'beta_schedule'     : 'sigmoid', # default
        # 'beta_schedule'     : 'cosine',
        'n_diffusion_step'  : 1000,
        # 'sampling_step'     : 500,
        'sampling_step'     : 20,


        ## dataset
        'loader'            : "data_loader.image_data_loader.Dataset1D",
        # 'dataset_path'      : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4142435161_13901k/voxel_images_w_multi_color_v1',
        'dataset_path'      : 'f:/home/user/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v2/sheetsander_cast_images_z_49',
        # 'image_size'        : 64, # simple model setting
        'image_size'        : 343,
        # 'grid_3dim'         : 16, # simple model setting
        'grid_3dim'         : 49,
        'is_shuffle'        : True,
        'horizontal_flip'   : False,
        'convert_image_to'  : None,


        ## serialization
        'logbase'           : 'logs',
        'prefix'            : 'diffusion_1d/',
        # 'tag'               : "dataset_2_12900k_v3",
        # 'tag'               : "dataset_4_12900k_v2",
        # 'tag'               : "dataset_45_12900k_v1",
        # 'tag'               : "dataset_41424351_12900k_v4",
        # 'tag'               : "dataset_4142435161_13900k_v7",
        # 'tag'               : "dataset_4142435161_13900k_v8",
        # 'tag'               : "dataset_4142435161_13901k_v2/",
        # 'tag'               : "dataset_41435161_13900k_v1",
        'tag'               : "H100_real_models_dataset_v2_1",


        'exp_name'          : watch(diffusion_1d_train_args_to_watch),

        ## training
        'trainer'           : 'trainer.diffusion_1d_trainer.Trainer1D',
        # 'batch_size'        : 64, # simple model setting
        'batch_size'        : 3, # simple model setting
        'learning_rate'     : 8e-5,
        'train_step'        : 800000,           # total training steps
        # 'gradient_accumulate_every' : 2,        # gradient accumulation steps # simple model setting
        'gradient_accumulate_every' : 30,
        'ema_decay'          : 0.995,           # exponential moving average decay
        'amp'                : True,            # turn on mixed precision
        'device'             :'cuda:0'
    },



    'conditional_image_diffusion': {
        # 'USER_NAME'         : "haxhi",
        'USER_NAME'         : "user",

        ## model
        # 'model'             : 'models.unet_2d.Unet',
        # 'model'             : 'models.unet_2d_simple_devel.Unet',
        'model'             : 'models.unet_2d_simple_devel2.Unet',

        'dim_mults'         : (1, 2, 4, 8),
        'flash_attn'        : True,
        'self_condition'    : False, # defalut = False

        'init_dim'          : 64,
        # 'init_dim'          : 128,


        # 'diffusion'         : 'models.conditional_image_diffusion.GaussianDiffusion',
        # 'diffusion'         : 'models.conditional_image_diffusion_simple_devel.GaussianDiffusion',
        # 'diffusion'         : 'models.conditional_image_diffusion_cfg_devel.GaussianDiffusion',
        'diffusion'         : 'models.conditional_image_diffusion_cfg_devel2.GaussianDiffusion',


        'beta_schedule'     : 'sigmoid', # default = 'sigmoid' ['sigmoid', 'cosine']
        'n_diffusion_step'  : 1000, # default = 1000
        'sampling_step'     : 20,


        ## dataset
        "loader":"data_loader.cond_image_data_loader.Cond_image_dataloader",
        "dataset_config": {"dataset" : {"name": "celeba",
                                        "path": "/home/dev/workspace/nedo-dismantling-PyBlender/dataset_4142435161_13900k/voxel_images_w_multi_color_v1",
                                        # "path": '/home/user/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v2/sheetsander_cast_images_x_49',
                                        # "path": '/home/user/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v2/sheetsander_cast_images_wo_body_x_49',
                                        # "path": '/home/user/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v2/sheetsander_cast_images_wo_body_z_49',
                                        # "path": '/home/user/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v2/sheetsander_cast_images_z_49',
                                        "min": -1,
                                        "max": 1,
                                        "h": 32, # only effects 'type=None or center'
                                        "type": "pattern",
                                        "p": 0.2} # only effects 'type=random'
                           },

        'image_size' : 64,
        # 'image_size' : 344,


        ## serialization
        'logbase'           : 'logs',
        'prefix'            : 'conditional_diffusion2/',
        'tag'               : "flower_image_v1",
        # 'tag'               : "dataset_13901k_v11",
        # 'tag'               : "H100_real_models_dataset_v2_18",
        # 'tag'               : "4090b_real_models_dataset_v2_tmp18_3",


        # v2_5 : data loader 64dimと同じ，　mask_label = (mask == 1).float()  # shape: (B, 1, H, W)　にして未観測部分のみのloss を取ろうとしたが逆
        # v2_6 : data loader 64dimと同じ，　mask_label = (mask == -1).float()  # shape: (B, 1, H, W)　にして未観測部分のみ(mask=-1が未観測，mask=1が観測済み)
        # v2_7 : data loader 64dimと同じ，　mask cond random を0.5にした lossの計算も全体に変更．v2_6はlossの計算をマスクの部分のみ．
        # v2_8 : data loader　ランダムサンプルする割合を5倍に．mask cond random を廃止. lossの計算も全体に変更．
        # v2_9 : data loader　ランダムサンプルする割合を5倍に．mask cond random を廃止. lossの計算も全体に変更． horizontal flip をtrueに
        # v2_10 : data loader　ランダムサンプルする割合を6倍に．mask cond random を廃止. lossの計算をマスク部分のみに． horizontal flip をtrueに
        # v2_11 : data loader　ランダムサンプルする割合を6倍に．mask cond random = 0.5 . lossの計算をマスク部分のみに． horizontal flip をtrueに
        # v2_12 : data loader　ランダムサンプルする割合を6倍に．mask cond random 廃止 . lossの計算をマスク部分のみに． horizontal flip をtrueに, binary mask cond mode　にネットワークを変更
        # dataset_13901k_v9: dataloader 64dim設定．mask cond random = None, loss マスク部分のみ, horizontal flip = True, binary mask cond style


        # "4090b_real_models_dataset_v2_2": 64dim設定．mask cond random なし， loss の計算をマスク部分のみに適用．horizontal flip true
        # "4090b_real_models_dataset_v2_3": 64dim設定．mask cond random=0.5， loss の計算をマスク部分のみに適用．horizontal flip true
        # "4090b_real_models_dataset_v2_4": 64dim設定．mask cond random なし， loss の計算をマスク部分のみに適用 あえてmask=1にして挙動確認．horizontal flip true
        # "4090b_real_models_dataset_v2_tmp2":今のbinarymask cond devel setting
        # "4090b_real_models_dataset_v2_tmp4":今のbinarymask cond devel setting cast image x ,lr = 8e-5
        # "4090b_real_models_dataset_v2_tmp6":今のbinarymask cond devel setting cast image x ,lr = 2e-5
        # "4090b_real_models_dataset_v2_tmp5":今のbinarymask cond devel setting cast image z ,lr = 2e-5 init dim  = 128
        # "4090b_real_models_dataset_v2_tmp7":今のbinarymask cond devel setting cast image z,lr = 8e-5  init_dim = 64


        'exp_name'          : watch(conditional_image_diffusion_train_args_to_watch),

        ## training
        'trainer'           : 'trainer.diffusion_conditional_image_trainer.Trainer',
        'batch_size'        : 96, # default
        # 'batch_size'        : 32,
        # 'batch_size'        : 18, # for 344
        'learning_rate'     : 8e-5, # default
        # 'learning_rate'     : 2e-5,
        # 'learning_rate'     : 9e-5,
        'train_step'            : 800000,           # total training steps
        # 'save_and_sample_every' : 2000,
        'save_and_sample_every' : 2000,
        'gradient_accumulate_every' : 2,        # gradient accumulation steps
        # 'gradient_accumulate_every' : 5,        # gradient accumulation steps #for 344
        'ema_decay'          : 0.995,           # exponential moving average decay
        'amp'                : True,            # turn on mixed precision
        'calculate_fid'      : False,
        'device'             :'cuda:0'
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
                                        "path": "/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_4142435161_13901k/voxel_images_w_multi_color_v1",
                                        # "path": '/home/user/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v2/sheetsander_cast_images_z_49',
                                        "min": -1,
                                        "max": 1,
                                        "h": 32,
                                        "type": "pattern",
                                        "p": 0.2}},
        'image_size' : 64,
        # 'image_size' : 344,


        ### model
        "model" : 'models.vaeac.vaeac.EncoderDecoder',
        "model_config": {"model": { "name": "EncoderDecoderNet",
                                    "last_layer": "tanh",
                                    "inp_channels": 3,
                                    "n_hidden": 32, # for 64dim
                                    # "n_hidden": 96, # for 344dim
                                    "fc_hidden": 100, # for 64dim
                                    # "fc_hidden": 300, # for 344dim
                                    "fc_out": 50, # for 64dim
                                    # "fc_out": 150,# for 344dim
                                    "optimizer": "adam",
                                    # "lr": 0.0001,# for 64dim
                                    "lr": 0.00005,# for 344dim
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
                            "batch_size": 96, # for 64dim
                            # "batch_size": 18, # for 344dim
                            'gradient_accumulate_every' : 2, # for 64dim
                            # 'gradient_accumulate_every' : 5, # for 344dim
                            "shuffle": True,
                            "step-log": 5,
                            'peek-validation': 10000}
                        },

        ## serialization
        'logbase'           : 'logs',
        'prefix'            : 'vaeac/',
        # 'tag'               : "image_align_celeba_s2",
        'tag'               : "H100_simple_models_dataset_13901k_v1",
        # 'tag'               : "H100_real_models_dataset_v2_2",
        'exp_name'          : watch(vaeac_train_args_to_watch),
    },




    'diffusion_plan': {
        # 'USER_NAME'         : "haxhi",
        'USER_NAME'         : "dev",

        # 'policy'            :'policy.cutting_surface_planner_v7.cutting_surface_planner',
        # 'policy'            :'policy.cutting_surface_planner_v8.cutting_surface_planner', # for real model
        'policy'            :'policy.cutting_surface_planner_v9.cutting_surface_planner', # for simple model
        'batch_size'        : 32,  # diffusion default
        # 'batch_size'        : 16, # for real model vaeac
        # 'batch_size'        : 18,
        # 'batch_size'        : 64,

        ## policy_config
        'policy_config'     :{
                                'ctrl_mode':"epsilon_greedy_00",
                                # 'ctrl_mode':"prior_based_ep_00",
                                # 'ctrl_mode':"epsilon_greedy_01",
                                # 'ctrl_mode':"epsilon_greedy_05",
                                # 'ctrl_mode':"random",
                                # 'ctrl_mode':"no_cond",
                                # 'ctrl_mode':"oracle_obs",
                                "image_mask_config_b" : {"target_mask"  :np.asarray([0.2,0.8,0.8]),
                                                        "target_mask_lb":np.asarray([0.2,0.8,0.8])-np.asarray([0.25,0.25,0.25]),
                                                        "target_mask_ub":np.asarray([0.2,0.8,0.8])+np.asarray([0.25,0.25,0.25])}, #np.asarray([0.7,0.2,0.2]
                                "image_mask_config_r": {"target_mask"   :np.asarray([0.8,0.2,0.2]),
                                                        "target_mask_lb":np.asarray([0.8,0.2,0.2])-np.asarray([0.1,0.1,0.1]),
                                                        "target_mask_ub":np.asarray([0.8,0.2,0.2])+np.asarray([0.2,0.2,0.2])}, #np.asarray([0.2,0.6,0.6])
                                "image_mask_config_y": {"target_mask"   :np.asarray([0.8,0.8,0.2]),
                                                        "target_mask_lb":np.asarray([0.8,0.8,0.2])-np.asarray([0.1,0.1,0.1]),
                                                        "target_mask_ub":np.asarray([0.8,0.8,0.2])+np.asarray([0.2,0.2,0.6])},
                                # "image_mask_config_b" : {"target_mask"  :np.asarray([0.0,0.0,1.0]),
                                #                         "target_mask_lb":np.asarray([0.0,0.0,1.0])-np.asarray([0.1,0.1,0.1]),
                                #                         "target_mask_ub":np.asarray([0.0,0.0,1.0])+np.asarray([0.1,0.1,0.0])}, #np.asarray([0.7,0.2,0.2]
                                # "image_mask_config_r": {"target_mask"   :np.asarray([1.0,0.0,0.0]),
                                #                         "target_mask_lb":np.asarray([1.0,0.0,0.0])-np.asarray([0.1,0.1,0.1]),
                                #                         "target_mask_ub":np.asarray([1.0,0.0,0.0])+np.asarray([0.0,0.1,0.1])}, #np.asarray([0.2,0.6,0.6])
                                # "image_mask_config_y": {"target_mask"   :np.asarray([0.0,1.0,0.0]),
                                #                         "target_mask_lb":np.asarray([0.0,1.0,0.0])-np.asarray([0.1,0.1,0.1]),
                                #                         "target_mask_ub":np.asarray([0.0,1.0,0.0])+np.asarray([0.1,0.0,0.1])},
                                # 'infer_model':"vaeac",
                                # 'infer_model':"diffusion",
                                'infer_model':"conditional_diffusion",
                                # 'infer_model':"diffusion_1D",
                                "decision_mode": {
                                                    # "remove_outliers_cal_cost_mean",
                                                    # "mode": "cal_cost_mean",
                                                    # "cal_cost_mode",
                                                    # "mode": "cal_cost_mode",  # "cal_cost_mean_ucb", # default
                                                    # "mode": "cal_cost_mean_ucb", # default
                                                    # "mode": "clip_ucb",
                                                    # "param":{
                                                    #         "ucb_lb":0.3  # 1.0 or 0.99 or
                                                    #          },
                                                    "mode": "clip_ucb_raw",
                                                    "param":{
                                                            "ucb_lb":0.5,  # 1.0 or 0.99 or
                                                            # "ucb_lb":1.0  # 1.0 or 0.99 or
                                                             }
                                                    },
                                "cfg_omega": 0.2,
                                },
        ## eval data loading
        ## 'eval_data_path'    : 'f:/home/{USER_NAME}/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/',
        ## 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/nedo_dismantling_dataset/dataset_1/geom_eval/',
        ## 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_1/geom_eval',
        ## 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_2_12900k/geom_test',
        ## 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test',
        ## 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_2',

        # 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_3',
        # 'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1',
        'eval_data_path'    : 'f:/home/{USER_NAME}/nedo-dismantling-PyBlender/dataset/dataset_4_12900k/geom_test_1',

        'eval_data_lists'    : {
                                # 'Object_1':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_3/Boxy_0",
                                # 'Object_2':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_3/Boxy_1",
                                # 'Object_3':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_3/Boxy_2",
                                # 'Object_4':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1/Boxy_0",
                                # 'Object_5':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1/Boxy_1",
                                # 'Object_6':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1/Boxy_2",
                                # 'Object_7':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1/Boxy_0",
                                # 'Object_8':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1/Boxy_1",
                                # 'Object_9':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1/Boxy_2",

                                'Object_1' : "/home/dev/workspace/nedo-dismantling-PyBlender/dataset_4_12900k/geom_test_3/Boxy_0",

                                },



        ##############################
        ## simple model prefix
        ###############################
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/point_e_diffusion/T1000_D64_dataset_41435161_13900k_v1/',
        ## 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_1d/T1000_D64_dataset_4142435161_13901k_v1/',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_1d/T1000_D64_dataset_4142435161_13901k_v3/',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/conditional_diffusion2/T1000_D64_dataset_13901k_v11',
        ## 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac/D64_H100_simple_models_dataset_13901k_v1/',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac/D64_H100_simple_models_dataset_13901k_v2/',
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac/D64_H100_simple_models_dataset_13901k_v3/',
        'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/conditional_diffusion2/T1000_D64_flower_image_v1/',


        # 'diffusion_epoch'   : '5000',
        # 'diffusion_epoch'   : '16000',
        # 'diffusion_epoch'   : '10000',
        # 'diffusion_epoch'   : '40000',
        # 'diffusion_epoch'   : '50000',
        'diffusion_epoch'   : '100000',
        # 'diffusion_epoch'   : '200000',
        # 'diffusion_epoch'   : '400000',
        'prefix'            : 'diffusion_plans/dataset_4142435161_13901k_v1_2/',

        ##############################
        ## real model prefix
        ###############################
        # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/conditional_diffusion2/T1000_D344_H100_real_models_dataset_v2_19',
        # # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac/D344_H100_real_models_dataset_v2_2',
        # # 'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/conditional_diffusion2/T1000_D64_4090b_real_models_dataset_v2_tmp18_2',
        # # 'diffusion_epoch'   : '200000',
        # # 'diffusion_epoch'   : '70000',
        # # 'diffusion_epoch'   : '64000',
        # # 'diffusion_epoch'   : '84000',
        # 'diffusion_epoch'   : '98000',
        # # 'diffusion_epoch'   : '32000',
        # 'prefix'            : 'diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/',



        'task_step' :8,
        # 'task_step' :10,
        # 'task_step' :15,

        # 'start_action_idx':19,
        # 'start_action_idx':24, # or 17 or 73 for SheetSander_kwon
        # 'start_action_idx':41, # for dataset_eval_4_3 -> action index = 41
        # 'start_action_idx':4,

        # "start_action_idx":{"Object_1":24,
        #                     "Object_2":73,
        #                     "Object_3":70},


        # "start_action_idx":{
        #                     "Object_1":24,
        #                     "Object_2":73,
        #                     "Object_3":70,
        #                     "Object_4":24,
        #                     "Object_5":127,
        #                     "Object_6":131,
        #                     },

        "start_action_idx":{
                            "obj_tag":"123456789",
                            "Object_1":np.arange(47,41-1,-1), #41
                            "Object_2":np.arange(47,41-1,-1), #41
                            "Object_3":np.arange(47,41-1,-1), #41
                            "Object_4":np.arange(0,4+1),   #4
                            "Object_5":np.arange(0,4+1),   #4
                            "Object_6":np.arange(0,4+1),   #4
                            "Object_7":np.arange(0,4+1),   #4
                            "Object_8":np.arange(0,4+1),   #4
                            "Object_9":np.arange(0,4+1),   #4
                            },

        ####################
        ## real model setting
        #####################
        # "start_action_idx":{
        #                     "obj_tag":"123456",
        #                     "Object_1":np.arange(48,24-1,-1), #24,
        #                     "Object_2":np.arange(97,73-1,-1),# 73,
        #                     "Object_3":np.arange(97,69-1,-1), # 69,
        #                     "Object_4":np.arange(48,24-1,-1), #24,
        #                     "Object_5":np.arange(98,126+1),   # 126
        #                     "Object_6":np.arange(97,73-1,-1), # 73
        #                     },



        ## serialization
        'logbase'           : 'logs',
        'suffix'            : 'f:{policy_config["ctrl_mode"]}',
        'observation_mode'  : 'partial_obs',
        # 'observation_mode'  : 'full_obs',
        # 'observation_mode'  : 'both_cs_obs',
        'iter'              : [0,6],
        # 'iter'              : [2,6],
        # 'iter'              : [0,2],
        # 'iter'              : [0,10],
        # 'tag'               : 'f:B{batch_size}_T{task_step}_fix_start_multi_step_partial_obs_a{start_action_idx}_{policy_config["infer_model"]}_{policy_config["decision_mode"]["mode"]}_v1_1',
        # 'tag'               : 'f:B{batch_size}_T{task_step}_fix_start_multi_step_{observation_mode}_a{start_action_idx}_{policy_config["infer_model"]}_{policy_config["decision_mode"]["mode"]}_v1_3',
        # 'tag'               : 'f:B{batch_size}_T{task_step}_fix_start_multi_step_{observation_mode}_a{start_action_idx}_{policy_config["infer_model"]}_{policy_config["decision_mode"]["mode"]}_v11_5',
        # 'tag'               : 'f:B{batch_size}_T{task_step}_{observation_mode}_a{start_action_idx["Object_1"]}a{start_action_idx["Object_2"]}a{start_action_idx["Object_3"]}_{policy_config["infer_model"]}_{policy_config["decision_mode"]["mode"]}_v11_7',
        # 'tag'               : 'f:B{batch_size}_T{task_step}_{observation_mode}_{policy_config["infer_model"]}_a{start_action_idx["obj_tag"]}_{policy_config["decision_mode"]["mode"]}_{policy_config["decision_mode"]["param"]["ucb_lb"]}_v13_2-----',
        'tag'               : 'f:B{batch_size}_T{task_step}_{observation_mode}_{policy_config["infer_model"]}_a{start_action_idx["obj_tag"]}_{policy_config["decision_mode"]["mode"]}_{policy_config["decision_mode"]["param"]["ucb_lb"]}_v12_1_for_paper_render',



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
