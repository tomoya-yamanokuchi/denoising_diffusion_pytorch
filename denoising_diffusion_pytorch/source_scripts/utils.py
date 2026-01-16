

import collections
import numpy as np
import torch
import pdb
import os
import collections
import importlib
import pickle
import os
import pickle
import glob
import torch
import pdb



DTYPE = torch.float
DEVICE = 'cuda:0'

#-----------------------------------------------------------------------------#
#------------------------------ numpy <--> torch -----------------------------#
#-----------------------------------------------------------------------------#

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

def import_class(_class):
    if type(_class) is not str: return _class
    ## 'diffusion' on standard installs
    repo_name = __name__.split('.')[0]
    ## eg, 'utils'
    module_name = '.'.join(_class.split('.')[:-1])
    ## eg, 'Renderer'
    class_name = _class.split('.')[-1]
    ## eg, 'diffusion.utils'
    module = importlib.import_module(f'{repo_name}.{module_name}')
    ## eg, diffusion.utils.Renderer
    _class = getattr(module, class_name)
    print(f'[ utils/config ] Imported {repo_name}.{module_name}:{class_name}')
    return _class

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
    states = glob.glob1(os.path.join(loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    # print(config)
    return config


# def load_config(loadpath):
#     config = pickle.load(open(loadpath, 'rb'))
#     print(f'[ utils/serialization ] Loaded config from {loadpath}')
#     # print(config)
#     return config

def save_config(obj,savepath):
    # savepath = os.path.join(*savepath)
    # pickle.dump(open(savepath, 'wb'))
    with open(savepath, mode="wb") as f:
        pickle.dump(obj, f)
    print(f'[ utils/config ] Saved config to: {savepath}\n')


def model_load(logdir,epoch,model):
        '''
            loads model
        '''
        loadpath = os.path.join(logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        step = data['step']
        model.load_state_dict(data['model'])

        return model


def load_model(loadpath):

    # import ipdb;ipdb.set_trace()

    # dataset_config      = load_config(*loadpath, 'dataset_config.pkl')
    model_config        = load_config(os.path.join(loadpath, 'model_config.pkl'))
    # model_config        = load_config(*loadpath, 'model_config.pkl')
    # trainer_config      = load_config(*loadpath, 'trainer_config.pkl')

    # trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    # dataset  = "hoge"
    # dataset   = dataset_config()
    model     = model_config
    # trainer   = trainer_config(model, dataset)


    epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    learned_model = model_load(loadpath,epoch,model)


    return learned_model