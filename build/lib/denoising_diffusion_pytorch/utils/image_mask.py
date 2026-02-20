
import torch


def bernoulli_mask(image_size):

        # p_val =  torch.FloatTensor(1).uniform_(0,1).item()
        p_val = 0.8
        bernoulli_mask = torch.bernoulli(torch.full((image_size,image_size), p_val))

        return bernoulli_mask