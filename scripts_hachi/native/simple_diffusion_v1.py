

import torch
from torch import nn

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np



from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils



print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    # mode            = "train" # train or eval
    mode            = "test" # train or eva

    image_dim = 64


    # dataset_path    = "./datasets/pidray/train/" # which includes *.png file
    # dataset_path    = "/home/haxhi/dataset/denoising_diffusion_pytorch/flower-image/train_concat/"
    # dataset_path    = "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/voxel_images"
    dataset_path    = "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/voxel_images_w_multi_color_v1"
    



    # save_folder     = "./results/x_ray_large"
    # save_folder     = "./results/flower_large_128"
    # save_folder     = "./results/flower_large"
    # save_folder     = f"./results/voxel_image_{image_dim}"
    save_folder     = f"./results/voxel_image_w_multi_color_v1_{image_dim}_v1"
    


    model_idx = 6      #idx model is loaded in test mode



    model = Unet(
        dim = image_dim,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    model.to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size = image_dim,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 500   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = Trainer(
        diffusion,
        dataset_path,
        train_batch_size = 64,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True,              # whether to calculate fid during training
        results_folder = save_folder,
    )





    if mode == "train":
        print(f"mode = train")
        trainer.train()



    elif mode == "test":
        print(f"mode = test")

        import numpy as np
        import os
        from PIL import Image,ImageDraw
        import pyvista as pv


        dataset_path        =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/Boxy_99"
        mesh_source         = f"{dataset_path}/blend/"
        sample_image_num    = 16


        test_save_folder            = save_folder +f"/eval_{model_idx}_hard_v2/"
        cond_save_path              = os.path.normpath(f"{test_save_folder}/voxel_images")
        create_folder(cond_save_path)

        ## hard
        slice_tag         = [3,4,5,6,7,8,9,10,11,12,14]

        ##  mid
        # slice_tag         = [3,6,9,10,14]

        ## soft
        # slice_tag         = [3,9,10]

        ## intermediate
        # slice_tag         = [0,2,3,6,7,9,11,13,15]

        ## ??
        # slice_tag         = [0,1,4,5,8,10,12,14,15]

        ## composite_1
        # slice_tag         = [0,3,4,7,8,11,12,15]
        ## composite_2
        # slice_tag         = [0,2,3,6,7,9,13,15]





        ## load inner boxes and merge
        path, f_name = get_path(mesh_source,".stl")
        mesh2 = pv.read(path[1])
        mesh3 = pv.read(path[2])
        mesh4 = pv.read(path[3])
        merged = mesh2.merge(mesh3)
        merged = merged.merge(mesh4)


        ## voxelize mesh and get sliced image
        s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
                            "side_length":16}


        box_array_handler   = pv_box_array(grid_config=s_grid_config)
        _                   = box_array_handler.cast_mesh_to_box_array(mesh=merged)
        box_arrays_data     = box_array_handler.get_box_array_data()


        nearby_cells = box_arrays_data.boxes
        colors       = box_arrays_data.colors
        centers      = box_arrays_data.grid_centers
        grid_2dim    = box_arrays_data.grid_2dim_size
        grid_3dim    = box_arrays_data.grid_3dim_size
        batch_image_map = box_array_handler.batch_image_map


        ## save image
        imgs_z = box_array_handler.get_box_color_to_2d_image(box_color=colors,permute="z")
        pil_image_ = Image.fromarray((imgs_z*255).astype(np.uint8))
        save_name = f"{cond_save_path}/cast_z_axis{0}.png"
        pil_image_.save(save_name)


        # slice_img = box_array_handler.get_slice_image(image=imgs_z,slice_tag=[3,4,6,12,14])
        slice_img = box_array_handler.get_slice_image(image=imgs_z,slice_tag=slice_tag)
        pil_image = Image.fromarray((slice_img*255).astype(np.uint8))
        save_name = f"{cond_save_path}/cast_z_axis_cond{0}.png"
        pil_image.save(save_name)


        img = slice_img
        numpy_image = np.asarray(img)

        normalizer = LimitsNormalizer(numpy_image)
        normalized_cond = normalizer.normalize(numpy_image).transpose(2,0,1)
        normalized_cond = to_torch(normalized_cond)

        cond ={0:{"idx":torch.where(normalized_cond>-1.0),
                    "val":normalized_cond}}


        trainer.load(model_idx)
        sample_image = trainer.model.sample(batch_size=sample_image_num,return_all_timesteps=True, cond = cond ).detach().cpu()
        batch_size, diffusion_step,image_dim,image_dim,channels = sample_image.shape


        # make denoising process gif
        batch_images = (torch.permute(sample_image,(0,1,3,4,2))*255.0).numpy().astype(np.uint8)
        for i in range(batch_size):
            img_save_path = os.path.normpath(f"{test_save_folder}/batch_{i}")
            create_folder(img_save_path)

            ims = []
            for j in range(diffusion_step):
                im =  Image.fromarray((torch.permute(sample_image[i][j],(1,2,0))*255.0).numpy().astype(np.uint8))
                ims.append(im.quantize())

            # import ipdb;ipdb.set_trace()

            save_name = img_save_path+f"/sample_{i}_diffusion.gif"
            ims[0].save(save_name, save_all=True, append_images=ims[1:], optimize=False, duration=50, loop=0)
            ims[-1].save(img_save_path+f"/sample_{i}.png")



            # aa = to_torch(np.asarray(pil_image_))
            # bb = to_torch(batch_images[i][-1])


            target_image_mini_batch = box_array_handler.get_2d_image_to_mini_batch_image(np.asarray(pil_image_),"x")
            sample_image_mini_batch = box_array_handler.get_2d_image_to_mini_batch_image(batch_images[i][-1],"x")

            mse_loss_fn = nn.MSELoss()
            total_loss = []
            for k in range(target_image_mini_batch.shape[0]):
                if k in slice_tag:
                    h=1
                else:
                    aa = to_torch(target_image_mini_batch[k])
                    bb = to_torch(sample_image_mini_batch[k])
                    loss = to_np(mse_loss_fn(aa,bb))
                    total_loss.append(loss)
                    # print(f"idx:{k} | loss:{loss}")
            loss = np.asarray(total_loss).mean()
            print(f"loss:{loss}")


            # import ipdb;ipdb.set_trace()
            # loss = nn.MSELoss()
            # loss =to_np( loss(aa,bb))

            data = {"denoising_process" : batch_images[i],
                    "loss"              : loss,
                    "slice_tag"         : slice_tag,
                    "s_grid_config"     : s_grid_config,
                    "gt_mesh"           : merged}
            pickle_utils().save(data,save_path=img_save_path+f"/denoising_process_{i}.pickle")
            del ims


        hoge = (torch.permute(sample_image[0][-2],(1,2,0))).numpy()
        hoge = hoge.clip(0,1,hoge)
        # hoge = np.random.rand(64,64,3)
        updated_colors = box_array_handler.cast_2d_image_to_box_color(image=hoge,permute="z")
        colors = updated_colors

        import ipdb;ipdb.set_trace()
        exit()


        # 表示
        plotter = pv.Plotter()

        for idx,elements in enumerate(nearby_cells):
            # if np.all(colors[int(elements)] != np.asarray([0,0,0])):
            if np.all(colors[int(elements)] <= np.asarray([0.9,0.9,0.9])):
                plotter.add_mesh(nearby_cells[elements], color = colors[int(elements)] ,opacity = 0.9 , show_edges=True)
            else:
                plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True, edge_opacity= 0.01)

            # plotter.add_mesh(nearby_cells[elements], color = colors[int(elements)] ,opacity = 0.1 , show_edges=True)

        plotter.add_points(centers, render_points_as_spheres=True, color = [0,0,0], opacity = 0.5, )


        # plotter.add_mesh(sgrid,scalars=colors, rgb=True,opacity = 0.5,show_edges=True)
        plotter.add_mesh(merged,color =[0.1,0.8,0.8], opacity =0.4)

        arrow_x = pv.Arrow(
        start=(0, 0, 0), direction=(1, 0, 0), scale=0.08)
        arrow_y = pv.Arrow(
        start=(0, 0, 0), direction=(0, 1, 0), scale=0.08)
        arrow_z = pv.Arrow(
        start=(0, 0, 0), direction=(0, 0, 1), scale=0.08)
        plotter.set_background('white')
        # plotter.add_camera_orientation_widget()
        plotter.add_mesh(arrow_x, color="r")
        plotter.add_mesh(arrow_y, color='g')
        plotter.add_mesh(arrow_z, color='b')
        plotter.show()