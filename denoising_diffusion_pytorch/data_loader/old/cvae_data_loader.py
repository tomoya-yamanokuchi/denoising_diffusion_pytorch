

from pathlib import Path
from functools import partial
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


from torchvision import transforms as T

from PIL import Image

from denoising_diffusion_pytorch.models.helpers import exists
from denoising_diffusion_pytorch.utils.image_mask import bernoulli_mask
from denoising_diffusion_pytorch.version import __version__



def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image




class cvaedataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):

        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        trans_formed_img = self.transform(img)
        # cond_image          = self.transform(img)
        cond_image       =  trans_formed_img.detach().clone()
        # p_val =  torch.FloatTensor(1).uniform_(0,1).item()
        # bernoulli_mask = torch.bernoulli(torch.full((self.image_size,self.image_size), p_val))
        # bernoulli_mask_ = bernoulli_mask(self.image_size)
        # cond_image[:,bernoulli_mask_==1] = 0.0

        cond_image       =  trans_formed_img[:1,:,:].detach().clone()
        bernoulli_mask_ = bernoulli_mask(self.image_size)
        cond_image[:,bernoulli_mask_==1] = 0.0

        return {"train_image":trans_formed_img,
                "test_image" :trans_formed_img,
                "cond_image" :cond_image}

        # return {"image"     :trans_formed_img,
        #         "observed"  :cond_image,
        #         "mask"      :bernoulli_mask_}