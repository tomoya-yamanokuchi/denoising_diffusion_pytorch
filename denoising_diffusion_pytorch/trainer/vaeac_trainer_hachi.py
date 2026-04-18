

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
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


import torchvision

from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy
from denoising_diffusion_pytorch.utils.vaeac_utils.vaeac_utils import init_weights,get_optimizers,get_losses,load_ckpt,get_schedulers,save_images,save_ckpt

import torch.nn.functional as F


def cycle(dl):
    while True:
        for data in dl:
            yield data


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)



class Trainer(object):
    def __init__(
        self,
        model,
        dataset,
        cfg,
    ):
        super().__init__()


        model    = init_weights(model, cfg.model_config)
        optim    = get_optimizers(model, cfg.model_config)
        self.loss_fn  = get_losses(cfg.model_config)
        self.model, self.optim, metadata = load_ckpt(model, optim, cfg.train_config)
        self.dataset = dataset
        self.config  = cfg
        self.gradient_accumulate_every = self.config.train_config["train"]["gradient_accumulate_every"]
        # Set up random seeds
        seed = np.random.randint(2**32)
        self.step = 0
        if metadata is not None:
            seed = metadata['seed']
            self.step = metadata['ckpt']

        # Get schedulers after getting checkpoints
        self.scheduler = get_schedulers(self.optim, cfg.model_config, self.step)
        # Print optimizer state
        print(self.optim)


        # Random seed according to what the saved model is
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


        train_batch_size = self.config.train_config["train"]["batch_size"]
        data_samples = len(self.dataset)
        train_size   = int(len(self.dataset) * 0.9)
        val_size     = data_samples - train_size
        # shuffleしてから分割してくれる.
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        self.train_dl = cycle(torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, num_workers=8, shuffle=True,drop_last=True))
        self.val_dl = cycle(torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=1, shuffle=True))

        # create summary writer save folder
        self.sw_savepath = os.path.join(self.config.savepath, f'summary_writer')
        os.makedirs(self.sw_savepath, exist_ok=True)
        # # Get loss file handle to dump logs to
        # if not os.path.exists(self.config.savepath):
        #     os.makedirs(self.config.savepath)
        # lossesfile = open(os.path.join(self.config.savepath, 'losses.txt'), 'a+')



    def load(self, milestone):
        accelerator = self.accelerator


    def train(self):

        # if self.step is not None:
        self.writer = SummaryWriter(log_dir=self.sw_savepath)

        with tqdm(initial = self.step, total = self.config.train_config["train"]["train_step"]) as pbar:

            while self.step < self.config.train_config["train"]["train_step"]:
                # self.model.train()
                # self.optim.zero_grad()
                # data = next(self.train_dl)

                # # Change to required device
                # for key, value in data.items():
                #     data[key] = Variable(value)
                #     data[key] = to_device(data[key])

                # # Get all outputs
                # outputs = self.model(data)
                # loss_val = self.loss_fn(outputs, data, self.config.model_config)



                # # Backward
                # loss_val.backward()
                # self.optim.step()

                # # Update schedulers
                # self.scheduler.step()

                # self.step +=1

                self.model.train()
                # accumulate gradients over multiple steps
                for _ in range(self.gradient_accumulate_every):

                    data = next(self.train_dl)

                    # Change to required device
                    for key, value in data.items():
                        data[key] = to_device(Variable(value))

                    # Get all outputs
                    outputs = self.model(data)
                    loss_val = self.loss_fn(outputs, data, self.config.model_config)

                    # Normalize loss for accumulation
                    loss_val = loss_val / self.gradient_accumulate_every
                    loss_val.backward()

                self.optim.step()
                self.scheduler.step()
                self.optim.zero_grad()
                self.step += 1

                pbar.set_description(f'loss: {loss_val:.4f}')


                # Log into the file after some epochs
                # if self.step % self.config.train_config['train']['step-log'] == 0:
                if self.config.train_config['train']['step-log'] is not None:
                    ## selossesfile.write('Epoch: {}, step: {}, loss: {}\n'.format(
                    ##     epoch, ckpt, loss_val.data.cpu().numpy()
                    ## ))
                                    # print it
                    # print('step: {}, loss: {}'.format(self.step, loss_val.data.cpu().numpy()))
                    self.writer.add_scalar("Train_loss",loss_val,self.step)

                if self.step % self.config.train_config["train"]['peek-validation'] == 0:
                # if self.step % 10 == 0:
                # if self.step >2:
                    self.model.eval()
                    with torch.no_grad():
                        val_data = next(self.val_dl)

                        for key, value in val_data.items():
                            val_data[key] = Variable(value)
                            val_data[key] = val_data[key].cuda()
                            val_data[key] = to_device(val_data[key])

                        # Get all outputs
                        outputs = self.model(val_data)
                        loss_val = self.loss_fn(outputs, val_data, self.config.model_config)

                        print('Validation loss: {}'.format(loss_val.data.cpu().numpy()))
                        self.writer.add_scalar("Valid_loss",loss_val.data,self.step)
                        # lossesfile.write('Validation loss: {}\n'.format(loss_val.data.cpu().numpy()))
                        save_images(val_data, outputs, self.config, self.step,suffix="",save_path=self.config.savepath)
                        # break
                    self.writer.add_images('output_image', outputs["out"][:,3:,:], self.step,  dataformats='NCHW')
                    self.writer.add_images('observation_image', val_data["observed"], self.step,  dataformats='NCHW')
                    self.writer.add_images('target_image', val_data["image"], self.step,  dataformats='NCHW')
                    self.model.train()
                    # Save checkpoint
                    save_ckpt((self.model, self.optim), self.config.train_config, self.step, 0 ,override=False, save_path=self.config.savepath)
                pbar.update(1)