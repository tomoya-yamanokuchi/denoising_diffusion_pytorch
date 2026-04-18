import os
import pickle
import glob
import torch
import pdb

from collections import namedtuple
from denoising_diffusion_pytorch.utils.vaeac_utils.vaeac_utils import load_ckpt,get_optimizers

DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')
VAEExperiment = namedtuple('VAE', 'dataset renderer model diffusion ema trainer epoch')
VAEACxperiment = namedtuple('VAEAC', 'dataset renderer model diffusion ema trainer epoch')



def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'model-*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('model-', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)

    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    # print(config)
    return config

def load_diffusion(*loadpath, epoch='latest', device='cuda:0'):


    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    # render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    model_config._device = device
    diffusion_config._device = device

    # dataset_cfg={"dataset" : {"name": "celeba",
    #                                     "path": "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/dataset_4142435161_13901k/voxel_images_w_multi_color_v1",
    #                                     # "path": '/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v2/sheetsander_cast_images_z_49',
    #                                     "min": -1,
    #                                     "max": 1,
    #                                     "h": 32,
    #                                     "type": "pattern",
    #                                     "p": 0.2}}

    dataset = dataset_config()
    # dataset.__init__(cfg =dataset_cfg,image_size =64)

    # dataset = None
    # renderer = render_config()
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')



    trainer.load(epoch)



    return DiffusionExperiment(dataset, "hoge", model, diffusion, trainer.ema, trainer, epoch)


def load_vae(*loadpath, epoch='latest',load_train_data=True, device='cuda:0'):

    if load_train_data is True:
        dataset_config      = load_config(*loadpath, 'dataset_config.pkl')
        dataset             = dataset_config()
    else:
        dataset   ="hoge"

    vae_config          = load_config(*loadpath, 'vae_config.pkl')
    model_config        = load_config(*loadpath, 'model_config.pkl')
    trainer_config      = load_config(*loadpath, 'trainer_config.pkl')

    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    renderer  = None
    vae       = vae_config()
    model     = model_config(vae)
    trainer   = trainer_config(model, dataset)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)


    return VAEExperiment(dataset, renderer, vae, model, trainer.model, trainer, epoch)



def load_vaeac(*loadpath, epoch='latest',load_train_data=True, device='cuda:0'):
    if load_train_data is True:
        dataset_config      = load_config(*loadpath, 'dataset_config.pkl')
        dataset             = dataset_config()
    else:
        dataset   ="hoge"

    model_config        = load_config(*loadpath, 'model_config.pkl')
    trainer_config      = load_config(*loadpath, 'trainer_config.pkl')

    model     = model_config()

    trainer   = trainer_config(model, dataset)


    trainer.config.train_config["pretrained"] = os.path.join(*loadpath,f"model_checkpoint_{epoch}.pt")

    optim = get_optimizers(model, trainer.config.model_config)
    model, _, metadata = load_ckpt(model, optim, trainer.config.train_config)
    trainer.step = epoch

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    renderer  = None


    return VAEExperiment(dataset, renderer, model, model, model, trainer, epoch)
