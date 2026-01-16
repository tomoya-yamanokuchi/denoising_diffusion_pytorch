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
        self.path = cfg['dataset']['path']
        self.height = cfg['dataset']['h']
        self.type = cfg['dataset'].get('type', None)
        self.p = cfg['dataset'].get('p', 1)
        self.image_size = image_size
        self.pattern_mask = self._get_pattern_mask()
        # assert (mode in ['train', 'val']), 'mode in {} should be train/val'.format(self.__name__)
        self._get_files()
        
        self.transform = T.Compose([
            T.Resize(image_size,interpolation=InterpolationMode.NEAREST),
            T.ToTensor()  # 最後に Tensor 化
        ])


    def _get_pattern_mask(self):
        """
        Generates a random pattern mask for the images.

        The pattern mask is created by generating a random image and resizing it to a large size,
        then thresholding it to create a binary mask.

        Returns:
            np.ndarray: A binary mask of shape (, ).
        """
        dim_scale = 6 # 344/64
        # Get the pattern mask
        image = np.random.rand(600, 600)
        if self.image_size == 344:
            image = cv2.resize(image, (10000*dim_scale, 10000*dim_scale), cv2.INTER_CUBIC) # for 344 dim setting
        else:    
            image = cv2.resize(image, (10000, 10000), cv2.INTER_CUBIC) # for 64 dim setting
        image = (image > 0.25).astype(float)# for 64 dim setting
        # image = (image > 0.45).astype(float)# for 343 dim setting # H100_real_models_dataset_v2_4
        return image


    def _get_pattern_sample(self):
        """
        Samples a patch of the pattern mask for use as a mask on the image.

        This method selects a random region from the precomputed `pattern_mask` of size `self.image_size` x `self.image_size`. 
        It ensures that the fraction of dropped pixels (i.e., pixels with a value of 0) is between 5% and 90%.

        The fraction of dropped pixels is calculated by taking the mean value of the mask.
        A mask is sampled repeatedly until the fraction of dropped pixels is within the acceptable range.

        Returns:
            np.ndarray: A 2D binary mask (with shape `image_size` x `image_size`) sampled from the pattern mask.
        """
        # Get a mask sampled from the pattern image
        # Fraction of pixels dropped, this value has to be between 20 and 30 percent
        frac = 0
        dim_scale = 6
        # while not (frac >= 0.2 and frac <= 0.999):
        # while not (frac >= 0.05 and frac <= 0.9): # for 64 dim setting
        while not (frac >= 0.05 and frac <= 0.9):  # for  343 dim setting # H100_real_models_dataset_v2_4
            if self.image_size == 344:
                y_coord, x_coord = np.random.randint((10000*dim_scale)-self.image_size, size=(2, )) # for 344 dim setting
            else:
                y_coord, x_coord = np.random.randint(10000-self.image_size, size=(2, )) # for 64 dim setting
            mask = self.pattern_mask[y_coord:y_coord+self.image_size, x_coord:x_coord+self.image_size]
            frac = 1 - (mask).mean()
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
        # Get the idx' valued item
        # First fetch the image, then get the mask
        # Return the final image
        filename = self.files[idx]
        image = cv2.imread(filename)
        assert (image is not None), filename

        # OpenCV: BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # NumPy → PIL.Image に変換
        image = Image.fromarray(image)

        # torchvision transforms を適用
        transformed_image = self.transform(image)

        # Tensor → NumPy へ変換
        transformed_image = transformed_image.permute(1, 2, 0).numpy()

        # RGB → BGR に戻す
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)*255.0

        # import ipdb;ipdb.set_trace()

        image_ = transformed_image

        if self.image_size != 344:
            image_ = self._RandomHorizontalFlip(image_=image_)


        # Convert image to right format (Crop and scale)
        # original_image = cv2.resize(image, (self.image_size, self.image_size))

        # image_ = self._RandomHorizontalFlip(image_=original_image)
        
        # image_2 = self._RandomVerticalFlip(image_=image_1)
        # image_ = self._RandomRotation90(image_=image_2)
        
        
        # image_ = original_image

        image = (image_[:, :, ::-1]/255.0)*2 - 1

        mask = self._get_mask(image)
        observed = (image_[:, :, ::-1]/255.0)*2 - 1
        observed[mask.squeeze()==1,:]=-1.0

        return {
            'image': torch.Tensor(image.transpose(2, 0, 1)),
            'mask' : torch.Tensor(mask.transpose(2, 0, 1)),
            'observed': torch.Tensor(observed.transpose(2, 0, 1)),
        }
