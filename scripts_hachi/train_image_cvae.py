


import os
import numpy as np


from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.config import Config


class Parser(parser):
    dataset: str = 'Image_diffusion_2D'
    config: str =  'config.vae'

#---------------------------------- setup ----------------------------------#


args = Parser().parse_args('cvae')

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
    savepath    = (args.savepath, 'model_config.pkl'),
    input_dim   = image_size,
    latent_dim  = args.latent_dim,
    image_channels = len(args.convert_image_to),
    cond_channels  = len(args.convert_image_to),
    device    = args.device,
)





cvae_config = Config(
    args.cvae,
    savepath = (args.savepath, 'cvae_config.pkl'),
    device   = args.device
)




trainer_config = Config(
    args.trainer,
    savepath            = (args.savepath, 'trainer_config.pkl'),
    train_batch_size    = args.batch_size,
    train_lr            = args.learning_rate,
    log_freq            = args.log_freq,
    sample_freq         = args.sample_freq,
    save_freq           = args.save_freq,
    label_freq          = args.label_freq,
    n_samples           = args.n_samples,
    results_folder      = args.savepath,
)



model       = model_config()
cvae        = cvae_config(model)
trainer     = trainer_config(model = cvae, dataset = dataset)


original_config_path = args.savepath
original_config_path = os.path.join(original_config_path,"original_configs_backup.py")
args.save_config_file(original_config_path)

print(args.savepath)






n_epochs = args.n_epoch

for i in range(n_epochs+1):
    # np.random.seed()
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train()



# import ipdb;ipdb.set_trace()

trainer.train()

# import ipdb;ipdb.set_trace()