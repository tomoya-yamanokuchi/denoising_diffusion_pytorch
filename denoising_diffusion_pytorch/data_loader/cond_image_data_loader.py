'''
CelebA Dataset for PyTorch
Author: Rohit Jena
'''
import os
from os.path import join, exists
import shutil
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Define the O1-O6 mask ranges here
# The image is of 64 x 64
O_MASKS = {
    'o1': [16, 35, 26, 57],
    'o2': [28, 35, 47, 57],
    'o3': [14, 49, 26, 36],
    'o4': [14, 33, 26, 36],
    'o5': [30, 49, 26, 36],
    'o6': [20, 43, 43, 61],
}


def divide_dataset_basic(path, fraction=0.85):
    '''
    Helper function for dividing the CelebA dataset into train or test sets

    Check if path exists or not
    and then create subsets accordingly

    The paper creates a 85-15 % split. I use the same split here
    Change the `fraction` if you want to change the split
    '''
    if exists(join(path, 'train')):
        print('Path {} exists'.format(path))
        return

    # Divide into train and val
    all_files = None
    # for root, _, all_files in os.walk(join(path, 'img_align_celeba')):
    for root, _, all_files in os.walk(join(path, '')):
        all_files = list(map(lambda x: join(root, x), all_files))
        break
    # import ipdb;ipdb.set_trace
    np.random.shuffle(all_files)

    # Divide the dataset
    # They report only validation scores, which is 15% of the dataset
    divide = int(len(all_files)*fraction)
    mode = {
        'train' : all_files[:divide],
        'val'   : all_files[divide:],
    }

    # Copy the files
    for key, files in mode.items():
        cur_path = join(path, key)
        os.makedirs(cur_path)
        for filename in files:
            shutil.copy(filename, cur_path)
            print('copied {}'.format(filename))
        print("Copied all {} files to {}".format(key, cur_path))


class Cond_image_dataloader(Dataset):
    """
    A dataset class for loading and processing images for vaeac.

    This dataset class supports various types of masks for image corruption, such as random masks, center masks,
    pattern-based masks, etc. It also supports image augmentation operations like horizontal flip, rotation, etc.

    Args:
        cfg (dict): A configuration dictionary containing dataset paths, mask settings, and other parameters.
        image_size (int): The target size (height/width) for image resizing.

    Attributes:
        path (str): The root directory of the dataset.
        height (int): The height of the region to be masked in images.
        type (str): The type of mask to apply (e.g., 'center', 'random', etc.) currently only 'pattern' is supported.
        p (float): The probability for random mask generation.
        image_size (int): The target size for image resizing.
        pattern_mask (self._get_pattern_mask): The pattern mask used for generating pattern-based masks.
        files (list): A list of image file paths to be used in the dataset.
    """

    def __init__(self, cfg, image_size):
        """
        Initializes the dataset with configuration settings and prepares the list of image files.

        Args:
            cfg (dict): Configuration dictionary containing dataset paths and settings.
            image_size (int): The target size for resizing the images.

        """
        super(Cond_image_dataloader, self).__init__()
        # self.mode = mode
        self.path         = cfg['dataset']['path']
        self.height       = cfg['dataset']['h']
        self.type         = cfg['dataset'].get('type', None)
        self.p            = cfg['dataset'].get('p', 1)
        # ---
        self.image_size   = image_size
        self.pattern_mask = self._get_pattern_mask()
        # assert (mode in ['train', 'val']), 'mode in {} should be train/val'.format(self.__name__)
        self._get_files()

        self.transform = T.Compose([
            T.Resize(image_size,interpolation=InterpolationMode.NEAREST),
            T.ToTensor()  # 最後に Tensor 化
        ])


    def _get_pattern_mask(self):
        """
        Generates a random pattern mask via tiling instead of resizing to huge arrays.
        Uses a small random image tiled to cover the sampling area, avoiding the 60K×60K allocation.

        Returns:
            np.ndarray: A binary mask large enough for random cropping.
        """
        # Generate small random pattern and tile it to a reasonable size
        # Instead of 600→60000 resize (~10GB), tile 600×600 to cover sampling needs
        base_size = 600
        image = np.random.rand(base_size, base_size)
        image = cv2.resize(image, (base_size * 4, base_size * 4), cv2.INTER_CUBIC)
        image = (image > 0.25).astype(np.float32)

        # Tile to cover enough area for random cropping
        # Need at least image_size extra in each direction for cropping
        target_size = self.image_size * 10  # reasonable pool for random crops
        n_tiles = (target_size // image.shape[0]) + 1
        image = np.tile(image, (n_tiles, n_tiles))
        return image


    def _get_pattern_sample(self):
        """
        Samples a patch of the pattern mask for use as a mask on the image.
        Ensures fraction of dropped pixels is between 5% and 90%.
        Bounded to max 100 attempts to prevent infinite loops.

        Returns:
            np.ndarray: A 2D binary mask of shape (image_size, image_size).
        """
        max_extent = self.pattern_mask.shape[0] - self.image_size
        for _ in range(100):
            y_coord, x_coord = np.random.randint(max_extent, size=(2,))
            mask = self.pattern_mask[y_coord:y_coord+self.image_size, x_coord:x_coord+self.image_size]
            frac = 1 - mask.mean()
            if 0.05 <= frac <= 0.9:
                return mask
        # fallback: return last sampled mask
        return mask


    def _get_files(self):
        # Get all image file paths from root path and store in a list
        self.files = []
        # if not exists(join(self.path, self.mode)):
        #     divide_dataset_basic(self.path)

        for root, _, files in os.walk(join(self.path)):
            self.files.extend(sorted(list(map(lambda x: join(root, x), files))))


    def __len__(self):
        # Return length
        assert self.files != [], 'Empty file list'
        return len(self.files)


    def _get_mask(self, image):
        """
        Generates a mask for a given image based on the specified mask type.

        Args:
            image (np.ndarray): The input image to generate a mask for.

        Returns:
            np.ndarray: The generated mask based on the mask type.
        """
        # Get mask given in the config by checking type
        if self.type is None:
            mask = np.ones(image.shape)[:, :, :1]
            x_start, y_start = np.random.randint(image.shape[0] - self.height, size=(2, ))
            width, height = self.height, self.height
            mask[y_start:y_start+height, x_start:x_start+width] = 0

        # Center mask, create a mask of height H * H from center
        elif self.type == 'center':
            mask = np.ones(image.shape)[:, :, :1]
            c_y, c_x = image.shape[0]//2, image.shape[1]//2
            mask[c_y - self.height//2 : c_y + self.height//2, \
                 c_x - self.height//2 : c_x + self.height//2] = 0

        # Random mask, drop pixels randomly
        elif self.type == 'random':
            # mask = np.random.rand(*image.shape)[:, :, :1]
            # mask = (mask < self.p).astype(float)
            pp = np.random.rand()
            pp = np.random.uniform(low=0.0, high=1.0)
            # pp = 0.9
            mask = np.random.binomial(1.,pp,(image.shape))[:,:,:1]
            mask = mask.astype(float)
            # import ipdb;ipdb.set_trace()


        # Half mask, randomly pick one from left, right top bottom
        elif self.type == 'half':
            mask = np.ones(image.shape)[:, :, :1]
            # Get which half is to be masked in case one is chosen
            # and then choose at random
            left_start, top_start = 32*np.random.randint(2, size=(2, ))
            go_left = np.random.rand() < 0.5
            if go_left:
                mask[:, left_start:left_start+32] = 0
            else:
                mask[top_start:top_start+32, :] = 0

        # Pattern mask, you got to sample from the pattern generated
        elif self.type == 'pattern':
            mask = self._get_pattern_sample()[:, :, None]

        # Else, one of the O1-O6 masks
        elif self.type in O_MASKS.keys():
            x_start, x_end, y_start, y_end = O_MASKS[self.type]
            mask = np.ones(image.shape)[:, :, :1]
            mask[y_start:y_end, x_start:x_end] = 0

        # rest are not implemented for now
        else:
            raise NotImplementedError
        return mask

    def _RandomHorizontalFlip(self,image_):
        is_fliplr = np.random.uniform(low=0.0, high=1.0) < 0.5

        if is_fliplr:
            image = np.fliplr(image_).copy()
        else:
            image = image_

        return image


    def _RandomVerticalFlip(self,image_):
        is_flip = np.random.uniform(low=0.0, high=1.0) < 0.5

        if is_flip:
            image = np.flipud(image_).copy()
        else:
            image = image_

        return image

    def _RandomRotation90(self,image_):
        i = np.random.randint(0,4)

        image = np.rot90(image_, i).copy()

        return image

    def __getitem__(self, idx):
        # Load as RGB directly via PIL (no BGR→RGB→PIL→Tensor→NumPy→BGR chain)
        filename = self.files[idx]
        image = Image.open(filename).convert('RGB')
        img_tensor = self.transform(image)  # (C, H, W) in [0, 1]

        # Optional horizontal flip (not for 344x344)
        if self.image_size != 344:
            if np.random.random() < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])

        # Normalize to [-1, 1]
        image_norm = img_tensor * 2 - 1  # (C, H, W)

        # Get mask and create observed image
        image_np = image_norm.permute(1, 2, 0).numpy()  # (H, W, C) for _get_mask
        mask = self._get_mask(image_np)

        observed = image_np.copy()
        observed[mask.squeeze() == 1, :] = -1.0

        return {
            'image': image_norm,
            'mask' : torch.from_numpy(mask.transpose(2, 0, 1)).float(),
            'observed': torch.from_numpy(observed.transpose(2, 0, 1)).float(),
        }
