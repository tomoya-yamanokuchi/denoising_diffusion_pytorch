


import os
import torch


from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.config import Config

torch.cuda.empty_cache()
torch.cuda.ipc_collect()


class Parser(parser):
    dataset: str = 'Image_diffusion_2D'
    config: str =  'config.vae'

#---------------------------------- setup ----------------------------------#


args = Parser().parse_args('diffusion_1d')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = Config(
    args.loader,
    savepath   = (args.savepath, 'dataset_config.pkl'),
    folder     = args.dataset_path,
    image_size = args.image_size,
    grid_3dim  = args.grid_3dim,
    is_shuffle = args.is_shuffle,
    augment_horizontal_flip = args.horizontal_flip,
    convert_image_to = args.convert_image_to,
)



dataset = dataset_config()
image_size        = dataset.image_size
channels, seq_len = dataset.__getitem__(1).shape


# import ipdb;ipdb.set_trace()

model_config = Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    dim       = args.dim,
    out_dim   = channels,
    dim_mults = args.dim_mults,
    channels  = channels,
    device    = args.device,
)





diffusion_config = Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    seq_length = seq_len,
    timesteps = args.n_diffusion_step,
    sampling_timesteps = args.sampling_step, # number of steps
    beta_schedule = args.beta_schedule,      # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
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
    results_folder            = args.savepath
)


# import ipdb;ipdb.set_trace()





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