

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


diffusion_plan_args_to_watch = [
    ('prefix', ''),
    ('diffusion_epoch', 'PT'),
    ('tag', ''),
]






base = {
    'diffusion': {
        'USER_NAME'         : "haxhi",
        ## model
        'model'             : 'models.unet_2d.Unet',
        'dim_mults'         : (1, 2, 4, 8),
        'flash_attn'        : True,

        'diffusion'         : 'models.diffusion.GaussianDiffusion',
        'n_diffusion_step'  : 1000,
        'sampling_step'     : 500,


        ## dataset
        'loader'            : "data_loader.image_data_loader.Dataset",
        'image_size'        : 64 ,
        # 'dataset_path'      : 'f:/home/{USER_NAME}/workspace/nedo-dismantling-PyBlender/dataset_1/voxel_images_w_multi_color_v1',
        'dataset_path'      : 'f:/home/haxhi/dataset/nedo_dismantling_dataset/dataset_2/voxel_images_w_multi_color_v1',
        'horizontal_flip'   : True,
        'convert_image_to'  : "RGB",

        ## serialization
        'logbase'           : 'logs',
        'prefix'            : 'diffusion/',
        'tag'               : "v0",
        'exp_name'          : watch(diffusion_train_args_to_watch),

        ## training
        'trainer'           : 'trainer.diffusion_trainer.Trainer',
        'batch_size'        : 64,
        'learning_rate'     : 8e-5,
        'train_step'        : 700000,           # total training steps
        'gradient_accumulate_every' : 2,        # gradient accumulation steps
        'ema_decay'          : 0.995,           # exponential moving average decay
        'amp'                : True,            # turn on mixed precision
        'calculate_fid'      : True,
        'device'             :'cuda'
    },



    'diffusion_plan': {
        'USER_NAME'         : "haxhi",

        'policy'            :'policy.cutting_surface_planner.cutting_surface_planner',
        'batch_size'        : 5,

        ## loading
        'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising-diffusion-pytorch/logs/Image_diffusion_2D/diffusion/T1000_D64_test_2',
        'diffusion_epoch'   : '18',

        ## serialization
        'logbase'           : 'logs',
        'prefix'            : 'diffusion_plans/',
        'suffix'            : 'epsilon_greedy_00',
        'iter'              : [0,6],
        'tag'               : "T1000_D64_test_5",
        'exp_name'          : watch(diffusion_plan_args_to_watch),

        'device': 'cuda',
    },

}







Reacher_v2 = {
}
