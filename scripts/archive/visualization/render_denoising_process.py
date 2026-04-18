

import ray
from tqdm import tqdm
from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.os_utils import save_json ,get_folder_name,create_folder,pickle_utils,get_path
from denoising_diffusion_pytorch.utils.voxel_render import pv_voxel_render,pv_voxel_render_parallel



# ray.init(log_to_driver=False,num_cpus=48) # for ros
ray.init(log_to_driver=False) # for ro
if __name__ == '__main__':

    class Parser(parser):
        dataset: str = 'Image_diffusion_2D'
        config: str =  'config.vae'

    args = Parser().parse_args('diffusion_plan')



    # dataset_path    =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/Boxy_99"
    # mesh_source     = f"{dataset_path}/blend/"

    # ## load inner boxes and merge
    # path, f_name = get_path(mesh_source,".stl")
    # mesh2 = pv.read(path[1])
    # mesh3 = pv.read(path[2])
    # mesh4 = pv.read(path[3])
    # merged = mesh2.merge(mesh3)
    # merged = merged.merge(mesh4)




    # save_folder     = f"./results/voxel_image_64"+f"/eval_{10}_hard_v2/"
    # save_folder     = f"./results/voxel_image_64"+f"/eval_{13}_soft_compound_1_2/"
    # save_folder     = f"./results/voxel_image_64"+f"/eval_{13}_hard_v2_2/"
    # save_folder     = f"./results/voxel_image_w_multi_color_v1_64_v1"+f"/eval_{6}_hard_v2/"
    save_folder       = args.savepath
    save_folders      = get_folder_name(save_folder)


    image_loss = []
    slice_tag  = []

    # for i in tqdm(range(len(save_folders))):
    for i in range(5):
    # i = 6
        # if save_folders[i].startswith(f"batch_{i}"):
        if save_folders[i].startswith("batch"):
            data_folder = save_folder+"/"+save_folders[i]
            vox_save_folder = data_folder+ "/3d_denoising"
            create_folder(vox_save_folder)


            load_data = pickle_utils().load(load_path=data_folder+f"/denoising_process_{i}.pickle")
            print(f"image_loss:{load_data['loss']}")
            sampled_images = load_data["denoising_process"]
            # gt_mesh        = load_data["gt_mesh"]
            s_grid_config = load_data["s_grid_config"]


            pv_voxel_render_parallel().render_voxel_denoising_v3(save_path=vox_save_folder,
                                                                    s_grind_config=s_grid_config,
                                                                    sample_images=sampled_images)

    import ipdb;ipdb.set_trace()
