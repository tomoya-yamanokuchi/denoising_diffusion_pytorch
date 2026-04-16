import math
from pathlib import Path
from multiprocessing import cpu_count

import torch
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

from denoising_diffusion_pytorch.version import __version__


from torchvision import transforms as T, utils
from torch.utils.tensorboard import SummaryWriter
# constants


from denoising_diffusion_pytorch.models.helpers import num_to_groups,has_int_squareroot,exists,default,cycle,normalize_to_neg_one_to_one,unnormalize_to_zero_to_one,identity,extract,linear_beta_schedule,cosine_beta_schedule,sigmoid_beta_schedule


class Trainer1D(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 800000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 5000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'bf16',
        split_batches = True,
        max_grad_norm = 1.
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        data_samples = len(dataset)
        train_size   = int(len(dataset) * 0.9)
        val_size     = data_samples - train_size
        # shuffleしてから分割してくれる.
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_dl = cycle(torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, num_workers=min(cpu_count(), 8),
            shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=2))
        val_dl = cycle(torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=1, shuffle=True))
        dl = self.accelerator.prepare(train_dl)
        self.dl = cycle(dl)

        # dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        # dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 8)
        self.grid_2dim = dataset.grid_2dim
        self.grid_3dim = dataset.grid_3dim



        # dl = self.accelerator.prepare(dl)
        # self.dl = cycle(dl)

        # optimizer (fused CUDA kernel for Ampere+)

        self.opt = AdamW(diffusion_model.parameters(), lr = train_lr, betas = adam_betas, fused = True)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.sw_dir = Path(results_folder+"/sw_dir/")
        self.sw_dir.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # compile the inner model for faster training (torch 2.0+)

        if hasattr(self.model, 'model'):
            self.model.model = torch.compile(self.model.model)
        else:
            self.model = torch.compile(self.model)

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            # if self.step==0:
            self.writer = SummaryWriter(log_dir=self.sw_dir)

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')
                self.writer.add_scalar("Train_loss",total_loss,self.step)

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad(set_to_none=True)

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    # if self.step != 0 and self.step % 100 == 0:
                    # if self.step != 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_samples = torch.cat(all_samples_list, dim = 0)
                        
                        # import ipdb;ipdb.set_trace()

                        all_samples_tp = torch.permute(all_samples,(0,2,1))
                        # all_samples_tp_index = all_samples_tp[:,:,:3]

                        all_samples_batch = torch.zeros(self.num_samples,3,self.grid_2dim,self.grid_2dim).to(device)

                        # data = next(self.dl).to(device)
                        # all_samples_tp =  torch.permute(data[:self.num_samples,:],(0,2,1))

                        for i in range(all_samples_batch.shape[0]):
                            # import ipdb;ipdb.set_trace()
                            all_samples_tp_index  = torch.round(all_samples_tp[:,:,:3]*(self.grid_3dim-1.0)).int()[i]
                            all_samples_tp_values = all_samples_tp[:,:,3:][i]
                            aa= torch.zeros(self.grid_3dim,self.grid_3dim,self.grid_3dim,3).to(device)


                            aa[all_samples_tp_index[:,0],all_samples_tp_index[:,1],all_samples_tp_index[:,2]]=all_samples_tp_values

                            # import ipdb;ipdb.set_trace()
                            # dd = aa.reshape(64,64,-1)
                            dd = self.get_slice_image(aa)
                            # import ipdb;ipdb.set_trace()

                            pp = dd[None,:,:,:]
                            samples = torch.permute(pp,(0,3,1,2))
                            all_samples_batch[i]=samples

                        utils.save_image(all_samples_batch, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.writer.add_images('my_image_batch', all_samples_batch, milestone,  dataformats='NCHW')
                        # torch.save(samples, str(self.results_folder / f'sample-{milestone}.png'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

    # def get_slice_image(self, image=None):
    #     dim, _, _, channel = image.shape

    #     grid_2dim    = int(dim*4)
    #     grid_3dim    = dim
    #     batch_img_len = int(grid_2dim/grid_3dim)
    #     # import ipdb;ipdb.set_trace()

    #     cast_image = torch.zeros((grid_2dim,grid_2dim,3)).to(self.device)
    #     # cast_image = torch.zeros((grid_2dim,grid_2dim,3))


    #     # import ipdb;ipdb.set_trace()

    #     k = 0
    #     for j in range(batch_img_len):
    #         for i in range(batch_img_len):
    #             cast_image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = image[k]
    #             # cast_image[i*grid_3dim:(i+1)*grid_3dim,j*grid_3dim:(j+1)*grid_3dim] = image[k]
    #             # if k in slice_tag:
    #             #     cast_image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
    #             # else:
    #             #     cast_image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]*0.0
    #             k = k+1

    #     return cast_image
    
    def get_slice_image(self, image=None):
        # image: (D, H, W, C)
        dim, _, _, channel = image.shape
        batch_img_len = int(math.sqrt(dim))  # e.g., 7 if dim=49

        # (D=dim*dim, H, W, C) → (batch_img_len, batch_img_len, H, W, C)
        image = image.view(batch_img_len, batch_img_len, *image.shape[1:])  # (by, bx, H, W, C)

        # → (by * H, bx * W, C)
        cast_image = image.permute(0, 2, 1, 3, 4).reshape(
            batch_img_len * image.shape[2], batch_img_len * image.shape[3], channel
        )

        return cast_image

    def get_1d_to_2d_images(self, all_samples):
        all_samples_tp = torch.permute(all_samples,(0,2,1))
        # all_samples_tp_index = all_samples_tp[:,:,:3]

        all_samples_batch = torch.zeros(all_samples.shape[0],3,self.grid_2dim,self.grid_2dim).to(self.device)
        # all_samples_batch = torch.zeros(all_samples.shape[0],3,self.grid_2dim,self.grid_2dim)
        # import ipdb;ipdb.set_trace()

        # data = next(self.dl).to(device)
        # all_samples_tp =  torch.permute(data[:self.num_samples,:],(0,2,1))

        for i in range(all_samples_batch.shape[0]):
            all_samples_tp_index  = torch.round(all_samples_tp[:,:,:3]*(self.grid_3dim-1.0)).int()[i]
            # all_samples_tp_index  = torch.round(all_samples_tp[:,:,:3]).int()[i]
            all_samples_tp_values = all_samples_tp[:,:,3:][i]
            aa= torch.zeros(self.grid_3dim,self.grid_3dim,self.grid_3dim,3).to(self.device)
            # aa= torch.zeros(self.grid_3dim,self.grid_3dim,self.grid_3dim,3)
            all_samples_tp_index = torch.clip(all_samples_tp_index,0,15)
            # import ipdb;ipdb.set_trace()


            aa[all_samples_tp_index[:,0],all_samples_tp_index[:,1],all_samples_tp_index[:,2]]=all_samples_tp_values

            # import ipdb;ipdb.set_trace()
            # dd = aa.reshape(64,64,-1)
            dd = self.get_slice_image(aa)
            # import ipdb;ipdb.set_trace()

            pp = dd[None,:,:,:]
            samples = torch.permute(pp,(0,3,1,2))
            all_samples_batch[i]=samples

        return all_samples_batch
