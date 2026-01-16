


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F





class VAE_Handler_2dim_conv(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def get_loss(self, image, cond):

        # import ipdb;ipdb.set_trace()

        image_ = image
        # image_ = image.detach().cpu()

        output_image, mu ,log_var,latent_vector = self.model(image_, cond) # perform training

        # print(f"input_min   :{image_.min()}         input_max :{image_.max()}")
        # print(f"output_min  :{output_image.min()}   output_max:{output_image.max()}")

        bce_loss                                = F.binary_cross_entropy(output_image, image_, reduction='sum') # calculate loss
        # KL_loss                                 = (- 0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp()))
        KL_loss                                 = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        train_loss = bce_loss + KL_loss
        # train_loss = 1e-4*bce_loss + 1e-5*KL_loss
        # train_loss = bce_loss
        # train_loss = bce_loss + KL_loss*1e-10  # default

        # import ipdb;ipdb.set_trace()
        # if output_image.isnan():

        # if KL_loss > 100.0:
        # if torch.any(train_loss.isnan()) or torch.any(train_loss.isinf()):
        if torch.any(output_image.isnan()) or torch.any(output_image.isinf()):
            import ipdb;ipdb.set_trace()
            train_loss = bce_loss

        return train_loss, {"shape_loss"    : bce_loss,
                            "KL_loss"       : KL_loss,
                            "latent_vector" :latent_vector}


    @torch.no_grad()
    def decode(self,latent_vec, cond):
        self.model.eval()
        output_points=self.model.sample(latent_vec, cond)
        # return reconstruct_points.transpose(1,2)
        return output_points


    # @torch.no_grad()
    # def shape_decode(self,latent_vec, cond):
    #     self.model.eval()
    #     output_points=self.model.decode(latent_vec, cond)
    #     # return reconstruct_points.transpose(1,2)
    #     return torch.reshape(output_points, (self.model.input_dim,self.model.input_dim))



    # @torch.no_grad()
    # def shape_encode(self,image):
    #     self.model.eval()

    #     input_image = image
    #     output_image, mu ,log_var,latent_vector = self.model(input_image,mode = "eval") # perform training

    #     bce_loss= F.binary_cross_entropy(output_image, input_image, reduction='sum') # calculate loss

    #     # import ipdb;ipdb.set_trace()
    #     # reconstruct_image = torch.reshape(output_image, (self.model.input_dim,self.model.input_dim))
    #     # reconstruct_image = torch.reshape(output_image,(image.shape[0],image.shape[1],image.shape[2]))
    #     reconstruct_image = output_image


    #     return reconstruct_image, {  "shape_error"              : bce_loss,
    #                                   "reconstructed_points"    : reconstruct_image,
    #                                   "gt_points"               : image,
    #                                   "latent_vector"           : latent_vector,
    #                                   "latent_vector/mu"        : mu,
    #                                   "latent_vector/log_var"   : log_var}

