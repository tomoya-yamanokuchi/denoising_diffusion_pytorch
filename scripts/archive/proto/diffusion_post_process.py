
import numpy as np
import json
from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.os_utils import save_json ,get_folder_name,create_folder,pickle_utils




if __name__ == '__main__':







    class Parser(parser):
        dataset: str = 'Image_diffusion_2D'
        config: str =  'config.vae'

    args = Parser().parse_args('diffusion_plan')




    # save_folder     = f"./results/voxel_image_64"+f"/eval_{13}_intermediate//"
    # save_folder     = f"./results/voxel_image_64"+f"/eval_{13}_soft_compound_1_2/"
    # save_folder     = f"./results/voxel_image_64"+f"/eval_{13}_mid_v2/"
    # save_folder     = f"./results/voxel_image_64"+f"/eval_{6}_hard_v2/"
    # save_folder     = f"./results/voxel_image_w_multi_color_v1_64_v1"+f"/eval_{6}_hard_v2/"
    save_folder =args.savepath





    save_folders = get_folder_name(save_folder)


    image_loss = []
    slice_tag  = []
    for i in range(len(save_folders)):
        if save_folders[i].startswith("batch"):
            load_data = pickle_utils().load(load_path=save_folder+"/"+save_folders[i]+f"/denoising_process_{i}.pickle")
            print(f"image_loss:{load_data['loss']}")
            image_loss.append(load_data["loss"])
            slice_tag = load_data["slice_tag"]


    image_loss_np = np.asarray(image_loss)

    data = {"image_loss":image_loss_np.tolist(),
            "image_loss_mean":image_loss_np.mean(),
            "image_loss_std" :image_loss_np.std(),
            "image_loss_var" :image_loss_np.var(),
            "slice_tag"      : slice_tag
            }


    import ipdb;ipdb.set_trace()
    save_name = save_folder+"/post_processed_data.json"
    # save_yaml(data=data,save_path=save_name)
    save_json(data=data,save_path=save_name) 