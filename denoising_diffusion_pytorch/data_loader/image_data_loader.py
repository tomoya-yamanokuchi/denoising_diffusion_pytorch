

from pathlib import Path
from functools import partial
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset as pytorch_Dataset
from torchvision.transforms import InterpolationMode

from torchvision import transforms as T

from PIL import Image

from denoising_diffusion_pytorch.models.helpers import exists
from denoising_diffusion_pytorch.version import __version__



def convert_image_to_fn(img_type, image):
    """
    Converts the input image to the specified mode (e.g., 'RGB', 'L') if it doesn't already match.

    Args:
        img_type (str): The target image mode (e.g., 'RGB', 'L').
        image (PIL.Image.Image): The input image to be converted.

    Returns:
        PIL.Image.Image: The image converted to the specified mode, or the original image if it already matches.
    """
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class Dataset(pytorch_Dataset):
    """
    Custom dataset class for loading and transforming images.

    This dataset class handles loading images from a specified folder, applying transformations such as resizing,
    cropping, and optional augmentations like random horizontal flipping. Additionally, it provides a corrupted version
    of the image for the conditional input to the model.

    Args:
        folder (str): Path to the folder containing the images.
        image_size (int): Desired output size of the images.
        exts (list, optional): List of image extensions to include in the dataset. Defaults to ['jpg', 'jpeg', 'png', 'tiff'].
        augment_horizontal_flip (bool, optional): Whether to apply random horizontal flipping to images. Defaults to False.
        convert_image_to (str, optional): The mode to which the images should be converted (e.g., 'RGB'). Defaults to None.

    Attributes:
        folder (str): Path to the image folder.
        image_size (int): Desired image size.
        paths (list): List of file paths for the images in the dataset.
        transform (torchvision.transforms.Compose): Transformations to be applied to each image.
    """
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        """
        Initializes the dataset by collecting image file paths and setting up the transformation pipeline.

        Args:
            folder (str): Path to the folder containing the images.
            image_size (int): Desired output size of the images.
            exts (list, optional): List of image extensions to include in the dataset. Defaults to ['jpg', 'jpeg', 'png', 'tiff'].
            augment_horizontal_flip (bool, optional): Whether to apply random horizontal flipping to images. Defaults to False.
            convert_image_to (str, optional): The mode to which the images should be converted (e.g., 'RGB'). Defaults to None.
        """
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            # T.Resize(image_size),
            T.Resize(image_size,interpolation=InterpolationMode.NEAREST),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.RandomVerticalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class Dataset1D(pytorch_Dataset):
    """
    A custom dataset class for handling 1D data ([position.x, position.y, position.z, R, G, B]) with optional transformations.

    Args:
        folder (str): Path to the directory containing the image files.
        image_size (int): The size of the image after transformation (assumed to be square).
        grid_3dim (int, optional): The size of the 3D grid (default: 16).
        is_shuffle (bool, optional): Whether to shuffle the data (default: True).
        exts (list of str, optional): List of file extensions to look for (default: ['jpg', 'jpeg', 'png', 'tiff']).
        augment_horizontal_flip (bool, optional): Whether to apply horizontal flip augmentation (default: False).
        convert_image_to (optional): A function to convert the image to a specific format (default: None).
    """
    def __init__(
        self,
        folder,
        image_size,
        grid_3dim = 16,
        is_shuffle = True,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        """
        Initializes the dataset by setting parameters, loading file paths, and defining the image transformation.

        Args:
            folder (str): Path to the folder containing the images.
            image_size (int): Size to which images will be resized.
            grid_3dim (int, optional): The size of the 3D grid (default: 16).
            is_shuffle (bool, optional): Whether to shuffle the dataset (default: True).
            exts (list of str, optional): List of allowed file extensions (default: ['jpg', 'jpeg', 'png', 'tiff']).
            augment_horizontal_flip (bool, optional): If True, applies random horizontal flipping (default: False).
            convert_image_to (optional): Function for converting images to a specified format (default: None).
        """
        super().__init__()
        self.grid_3dim = grid_3dim
        self.is_shuffle  = is_shuffle
        self.grid_2dim = int(np.sqrt(grid_3dim)*grid_3dim)
        self.folder = folder
        self.image_size = image_size
        self.image_size_flatten  = int(image_size*image_size)
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size,interpolation=InterpolationMode.NEAREST),
            # T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.RandomVerticalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])


    # def get_2d_image_to_mini_batch_image(self,image,permute):
    #     """
    #     Revert sliced by arbitlary axis and tiled image self.grid_2dim, self.grid_2dim, 3) to minibatch image (self.grid_3dim, self.grid_3dim, self.grid_3dim, 3)  format.

    #     Args:
    #         image (torch.Tensor): The input image tensor (self.grid_2dim, self.grid_2dim, 3).
    #         permute (str): Specifies how to permute the mini-batch ('z', 'y', or 'x').

    #     Returns:
    #         batch_2d_image(torch.Tensor): A tensor representing the mini-batch of images in a grid, permuted as specified. size (self.grid_3dim, self.grid_3dim, self.grid_3dim, 3)
    #     """

    #     batch_img_len = int(self.grid_2dim/self.grid_3dim)

    #     batch_2d_image_ = torch.zeros((self.grid_3dim, self.grid_3dim, self.grid_3dim, 3))
    #     k = 0
    #     for j in range(batch_img_len):
    #         for i in range(batch_img_len):
    #             batch_2d_image_[k] = image[j*self.grid_3dim:(j+1)*self.grid_3dim,i*self.grid_3dim:(i+1)*self.grid_3dim]
    #             k = k+1

    #     if permute == "z":
    #         batch_2d_image  = batch_2d_image_
    #     elif permute == "y":
    #         batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
    #     elif permute == "x":
    #         batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)

    #     return batch_2d_image

    def get_2d_image_to_mini_batch_image(self, image, permute):
        # grid サイズ
        grid_2dim = self.grid_2dim
        patch_size = self.grid_3dim

        # [H, W, C] → [C, H, W]
        image = image.permute(2, 0, 1)  # 例: [3, 343, 343]

        # unfold を使って2次元にパッチを抽出
        patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        # → [C, num_patches_H, num_patches_W, patch_H, patch_W]

        # 次元を整理：[C, num_patches_H, num_patches_W, patch_H, patch_W] → [num_patches, patch_H, patch_W, c]
        patches = patches.contiguous().view(3, -1, patch_size, patch_size).permute(1, 2, 3, 0 )


        if permute == "z":
            batch_2d_image  = patches
        else:
        # elif permute == "y":
        #     batch_2d_image  = patches.transpose(1,0,2,3)
        # elif permute == "x":
        #     batch_2d_image  = patches.transpose(2,1,0,3)

        return batch_2d_image

    def generate_3d_indices(self,mini_batch_dim):
        r = torch.arange(mini_batch_dim)
        zz, yy, xx = torch.meshgrid(r, r, r, indexing='ij')  # shape: [D, D, D]
        indices = torch.stack([zz, yy, xx], dim=-1)  # shape: [D, D, D, 3]
        return indices.reshape(-1, 3)  # → [D³, 3]


    def __len__(self):
        return len(self.paths)


    def  __getitem__(self, index):
        """
        Retrieves the image at the specified index and processes it into a 1D data format (n, 6)-> (n,(position.x, position.y, position.z, R, G, B)).

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            torch.Tensor: converted tensor  (n, 6)-> (n,(position.x, position.y, position.z, R, G, B)) from image with shuffled indices if specified.
        """

    # def  getitems(self, index):
        path        = self.paths[index]
        img         = Image.open(path)
        torch_img   = self.transform(img)

        torch_img_tp        = torch.permute(torch_img,(1,2,0))
        mini_batch_image    = self.get_2d_image_to_mini_batch_image(torch_img_tp,"z")
        mini_batch_dim = mini_batch_image.shape[0]

        # インデックスを作成 (0から15の範囲)
        # indices = torch.tensor([[i, j, k] for i in range(mini_batch_dim) for j in range(mini_batch_dim) for k in range(mini_batch_dim)])  # shape: [4096, 3]
        indices = self.generate_3d_indices(mini_batch_dim=mini_batch_dim)
        values  = mini_batch_image[indices[:, 0], indices[:, 1], indices[:, 2]]
        result  = torch.cat((indices/(mini_batch_dim-1.0), values), dim=1)
        # result  = torch.cat((indices, values), dim=1)

        if self.is_shuffle is True:
            result_  =result[torch.randperm(result.size(0))]
            result_tp = torch.permute(result_,(1,0))
        else:
            result_tp = torch.permute(result,(1,0))

        return result_tp