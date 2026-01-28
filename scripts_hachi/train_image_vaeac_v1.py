


import os
import numpy as np


from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.config import Config


class Parser(parser):
    dataset: str = 'Image_diffusion_2D'
    config: str =  'config.vae'

#---------------------------------- setup ----------------------------------#


args = Parser().parse_args('vaeac')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#
dataset_config = Config(
    args.loader,
    savepath   = (args.savepath, 'dataset_config.pkl'),
    cfg        = args.dataset_config,
    image_size  = args.image_size
)

dataset     = dataset_config()
image_size  = dataset.image_size

model_config = Config(
    args.model,
    savepath    = (args.savepath, 'model_config.pkl'),
    cfg       = args.model_config,
    device    = args.device,
)



trainer_config = Config(
    args.trainer,
    savepath            = (args.savepath, 'trainer_config.pkl'),
    cfg                 = args
)


import ipdb; ipdb.set_trace()

model       = model_config()


trainer     = trainer_config(model = model, dataset = dataset)
                            #  savepath=(args.savepath, 'trainer_config.pkl'))

trainer.train()
# import ipdb;ipdb.set_trace()
