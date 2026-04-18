


import os



from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.config import Config


class Parser(parser):
    dataset: str = 'Image_diffusion_2D'
    config: str =  'config.vae'

#---------------------------------- setup ----------------------------------#


args = Parser().parse_args('diffusion')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = Config(
    args.loader,
    savepath   = (args.savepath, 'dataset_config.pkl'),
    folder     = args.dataset_path,
    image_size = args.image_size,
    augment_horizontal_flip = args.horizontal_flip,
    convert_image_to = args.convert_image_to
)



dataset = dataset_config()
image_size = dataset.image_size



model_config = Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    # dim       = image_size,
    dim       = 128, ## changed for H100 with 512 dim #default 64
    # dim       = 64, ## changed for H100 with 512 dim #default 64
    dim_mults = args.dim_mults,
    flash_attn = args.flash_attn,
    self_condition = args.self_condition,
    device    = args.device,
)





diffusion_config = Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    image_size = image_size,
    timesteps  = args.n_diffusion_step,            # number of steps
    sampling_timesteps = args.sampling_step,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    beta_schedule = args.beta_schedule,
    device = args.device
)




trainer_config  = Config(
    args.trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size = args.batch_size,
    train_lr         = args.learning_rate,
    train_num_steps  = args.train_step,         # total training steps
    gradient_accumulate_every = args.gradient_accumulate_every,    # gradient accumulation steps
    ema_decay                 = args.ema_decay,                # exponential moving average decay
    amp                       = args.amp,                       # turn on mixed precision
    calculate_fid             = args.calculate_fid,
    results_folder            = args.savepath
)




model       = model_config()
diffusion   = diffusion_config(model)
trainer     = trainer_config(diffusion_model = diffusion, dataset = dataset)


original_config_path = args.savepath
original_config_path = os.path.join(original_config_path,"original_configs_backup.py")
args.save_config_file(original_config_path)

print(args.savepath)

# import ipdb;ipdb.set_trace()

trainer.train()

# import ipdb;ipdb.set_trace()