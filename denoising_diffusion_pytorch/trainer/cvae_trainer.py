import os
import copy
import numpy as np
import torch
import einops
import pdb
import pickle
import gc
from tqdm import tqdm
from PIL import Image

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


import torchvision

from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy

import torch.nn.functional as F


def cycle(dl):
    while True:
        for data in dl:
            yield data



def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



class Trainer(object):
    def __init__(
        self,
        model,
        dataset,
        train_batch_size=32,
        train_lr=2e-5,
        ema_decay=0.995,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=10,
        save_freq=100,
        label_freq=10000,
        save_parallel=False,
        results_folder='./results',
        n_samples=2,
        device = "cuda",
        bucket=None,
    ):
        super().__init__()


        self.model = model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset



        # self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset, batch_size=train_batch_size, num_workers=8, shuffle=True, worker_init_fn=worker_init_fn
        # )

        # self.dataloader_vis = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=1, num_workers=1, shuffle=True, worker_init_fn=worker_init_fn
        # ))


        # self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset, batch_size=train_batch_size, num_workers=8, shuffle=False,drop_last=True)

        # self.dataloader_vis = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=1, num_workers=1, shuffle=True))


        data_samples = len(self.dataset)
        train_size = int(len(self.dataset) * 0.9)
        val_size = data_samples - train_size

        # shuffleしてから分割してくれる.
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])


        self.dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, num_workers=8, shuffle=False,drop_last=True)

        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=1, shuffle=True))


        self.logdir     = results_folder
        self.val_dir    = results_folder+"/valid/"
        self.sw_dir     = results_folder+"/runs/"
        self.bucket     = bucket
        self.device     = device
        self.n_samples  = n_samples

        os.makedirs(self.sw_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr , betas=(0.9, 0.999), weight_decay=0.001)

        self.reset_parameters()
        self.step = 0


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)


    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self,n_train_steps=None):


        if self.step==0:
            self.writer = SummaryWriter(log_dir=self.sw_dir)


        for i, data in enumerate(self.dataloader):
            self.model.train()

            if i  == 0:
                import ipdb;ipdb.set_trace()

            train_image  = data["train_image"].to(self.device)
            cond         = data["cond_image"].to(self.device)

            self.optimizer.zero_grad()   # Reset the gradients

            ## infer model
            train_loss, info    = self.model.get_loss(train_image, cond)

            reconstruction_loss = info["shape_loss"]
            KL_loss             = info["KL_loss"]

            # train_loss          = train_loss / self.gradient_accumulate_every
            # reconstruction_loss = reconstruction_loss / self.gradient_accumulate_every
            # KL_loss             = KL_loss/ self.gradient_accumulate_every

            # train_loss          = train_loss
            # reconstruction_loss = reconstruction_loss
            # KL_loss             = KL_loss

            train_loss.backward()
            self.optimizer.step()        # Update the weights and biases


            # print(f"  tain_loss: {train_loss :.2f} | Reconstruct_loss: {reconstruction_loss :.2f} |  KL_loss: {KL_loss :.4f}")


            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:

                loss_info={ "step"               : self.step,
                            "loss"               : to_np(train_loss),
                            "shape_loss"         : to_np(reconstruction_loss),
                            "kl_loss"            : to_np(KL_loss),
                            "infos": "hoge",
                            }

                print(f" Step:{self.step}| Train_loss : {loss_info['loss']:.10f}| Shape_loss: {loss_info['shape_loss'] :.10f}| KL_loss: {loss_info['kl_loss'] :.4f}|")
                print(f" Step:{self.step}| Laten_vec | mean: {to_np(info['latent_vector']).mean():.6f} var: {to_np(info['latent_vector']).var(0).mean():.6f}")
                # print(f" Step:{self.step}| Laten_vec | {to_np(info['latent_vector'])[0]}")
                self.writer.add_scalar("Train_loss",loss_info['loss'],self.step)
                self.writer.add_scalar("Shape_loss",loss_info['shape_loss'],self.step)
                self.writer.add_scalar("KL_loss",loss_info['kl_loss'],self.step)


            if self.sample_freq and self.step % self.sample_freq == 0:
                self.validate(n_samples=self.n_samples)

            self.step += 1


    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step' : self.step,
            'model': self.model.state_dict(),
            'ema'  : self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')



    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#


    def validate(self, n_samples=2):

        savepath = os.path.join(self.val_dir, f'valid-{self.step}')
        os.makedirs(savepath, exist_ok=True)


        valid_data = {}
        valid_loss = []

        target_image_l  = []
        cond_image_l    = []
        reconstruct_image_l = []
        for i in range(n_samples):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()


            target_image = batch["test_image"].to(self.device)
            cond_image   = batch["cond_image"].to(self.device)
            latent_feat  = torch.normal(0, 1, size=(cond_image.shape[0], self.model.model.n_latent_features)).to(self.device)


            # encode-decode
            reconstructed_image    = self.model.decode(latent_feat,cond_image)


            bce_loss= F.binary_cross_entropy(reconstructed_image, target_image, reduction='sum')

            infos = {  "reconstruct_error":bce_loss,
                       "reconstruct_image":reconstructed_image}

            valid_loss.append(to_np(infos["reconstruct_error"]))

            # cond_image_             = to_np(torch.permute(cond_image,(0,2,3,1)))[0]*255.0
            # reconstructed_image_    = to_np(torch.permute(reconstructed_image,(0,2,3,1)))[0]*255.0
            # target_image_           = to_np(torch.permute(target_image,(0,2,3,1)))[0]*255.0

            target_image_l.append(target_image)
            cond_image_l.append(cond_image)
            reconstruct_image_l.append(reconstructed_image)


            # # pil_img = Image.fromarray(reconstructed_image_.astype(np.uint8))
            # img_save_path_reconstruct = os.path.join(savepath, f'valid-{self.step}-{i}-reconstruct.png')
            # # pil_img.save(img_save_path)

            # # pil_img = Image.fromarray(target_image_.astype(np.uint8))
            # img_save_path_target = os.path.join(savepath, f'valid-{self.step}-{i}-target.png')
            # # pil_img.save(img_save_path)

            # # pil_img = Image.fromarray(cond_image_.astype(np.uint8))
            # img_save_path_cond = os.path.join(savepath, f'valid-{self.step}-{i}-cond.png')
            # # pil_img.save(img_save_path)


            img_save_path_reconstruct   = os.path.join(savepath, f'valid_{n_samples}_reconstruct.png')
            img_save_path_target        = os.path.join(savepath, f'valid_{n_samples}_target.png')
            img_save_path_cond          = os.path.join(savepath, f'valid_{n_samples}_cond.png')

        torchvision.utils.save_image(torch.cat(target_image_l,dim = 0), img_save_path_target, nrow = int(np.sqrt(n_samples)))
        torchvision.utils.save_image(torch.cat(cond_image_l,dim = 0), img_save_path_cond, nrow = int(np.sqrt(n_samples)))
        torchvision.utils.save_image(torch.cat(reconstruct_image_l,dim = 0), img_save_path_reconstruct, nrow = int(np.sqrt(n_samples)))

        print(f" Step:{self.step} | Validation Loss: {np.asarray(valid_loss).mean():.10f}")
        # Please note that validation is performed by deterministic mean
        # print(f" Step:{self.step} | Laten_vec | mean: {to_np(infos['latent_vector']).mean():.6f}")
        # print(f" Step:{self.step} | Laten_vec | var : {to_np(infos['latent_vector']).var():.6f}")
        # print(f" Step:{self.step} | Laten_vec | {to_np(infos['latent_vector'])}")
        self.writer.add_scalar("Valid_loss",np.asarray(valid_loss).mean(),self.step)
        self.writer.add_images('cond_image', torch.cat(cond_image_l,dim = 0) , self.step,  dataformats='NCHW')
        self.writer.add_images('reconstruct_image', torch.cat(reconstruct_image_l,dim = 0) , self.step,  dataformats='NCHW')
        self.writer.add_images('target_image', torch.cat(target_image_l,dim = 0) , self.step,  dataformats='NCHW')



