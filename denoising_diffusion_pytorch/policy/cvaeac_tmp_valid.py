'''
trainer.py : Contains train and validation loops
Author: Rohit Jena
'''
import os
import json

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# from utils import utils
from denoising_diffusion_pytorch.utils.vaeac_utils.vaeac_utils import get_model,get_data_loaders,get_optimizers,load_ckpt,get_losses,repeat_data,init_weights


# def validate(model= None, data = None):

#     outputs = model(data)

#     N, _, H, W = outputs['out'].shape

#     return outputs["out"][:,3:,]

# def load_vaeac_model():
#     cfg = {'dataset': {'name': 'celeba',
#                         'path': '/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/voxel_images_w_multi_color_v1',
#                         'min': -1,
#                         'max': 1,
#                         'h': 32,
#                         'type': 'pattern',
#                         'p': 0.2},
#             'train': {'num-epochs': 1000,
#                         'save-freq': 2000,
#                         'batch-size': 64,
#                         'shuffle': True,
#                         'step-log': 100},
#             'val': {'batch-size': 1,
#                     'num-samples': 8,
#                     'shuffle': True,
#                     'save-img': True},
#             'model': {'name': 'encoderdecodernet',
#                     'last_layer': 'tanh',
#                     'inp_channels': 3,
#                     'n_hidden': 32,
#                     'fc_hidden': 100,
#                     'fc_out': 50,
#                     'optimizer': 'adam',
#                     'lr': 0.0001,
#                     'beta1': 0.9,
#                     'beta2': 0.999,
#                     'scheduler': 'step',
#                     'decay-steps': 500,
#                     'decay-factor': 0.995,
#                     'weight-decay': 1e-05,
#                     'init': 'orthogonal',
#                     'loss': 'mse',
#                     'lr-override': False},
#             'reg': {'lambda_kl': 1,
#                     'lambda_reg': 1,
#                     'sigma_m': 10000,
#                     'sigma_s': 0.0001,
#                     'apply_reg': True},
#             'use-cuda': True,
#             # 'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_5_12900k_7/model_checkpoint_266000.pt',
#             # 'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_45_12900k_1/model_checkpoint_266000.pt',
#             # 'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_41424351_12900k1/model_checkpoint_80000.pt',
#             # 'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_41424351_12900k1/model_checkpoint_1062000.pt',
#             # 'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_41424351_12900k2/model_checkpoint_80000.pt',
#             # 'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_4142435161_13900k_v1/model_checkpoint_120000.pt',
#             # 'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_4142435161_13900k_v4/model_checkpoint_120000.pt',
#             # 'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_4142435161_13900k_v5/model_checkpoint_120000.pt',
#             # 'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_4142435161_13900k_v5/model_checkpoint_200000.pt',
#             # 'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_4142435161_13900k_v6/model_checkpoint_1062000.pt',
#             # 'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_4142435161_13900k_v6/model_checkpoint_600000.pt',
#             'pretrained': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_4142435161_13901k_v1/model_checkpoint_1000000.pt',
#             'save-path': '/home/haxhi/workspace/vaeac/logs/saved_models/dataset_5_12900k_7/',
#             'peek-validation': 500}

#     '''
#     Main loop for validation, load the dataset, model, and
#     other things. Run validation on the validation set
#     '''
#     print(json.dumps(cfg, sort_keys=True, indent=4))

#     use_cuda = cfg['use-cuda']
#     _, _, _, val_dl = get_data_loaders(cfg)

#     model = get_model(cfg)
#     if use_cuda:
#         model = model.cuda()
#     # model = init_weights(model, cfg)
#     model.eval()

#     # Get pretrained models, optimizers and loss functions
#     optim = get_optimizers(model, cfg)
#     model, _, metadata = load_ckpt(model, optim, cfg)
#     loss_fn = get_losses(cfg)

#     # Set up random seeds
#     if metadata is not None:
#         seed = metadata['seed']
#     # Validation code, reproducibility is required
#     seed = 42

#     # Random seed according to what the saved model is
#     # np.random.seed(seed)
#     # torch.manual_seed(seed)
#     # if torch.cuda.is_available():
#     #     torch.cuda.manual_seed_all(seed)
    
    # return model