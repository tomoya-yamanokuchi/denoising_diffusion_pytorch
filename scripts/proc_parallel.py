import subprocess


def main():
    max_process = 10  # num of CPU cores (not threads)

    for package_name in ["multi_models"]:

        for axis_name in ["z"]:
            proc_list = []
            task_list = [str(i) for i in range(60)] ## total file name /100
            while task_list or proc_list:
                if len(proc_list) < max_process:
                    if task_list:
                        proc = subprocess.Popen(['python3', './scripts/resize_imgs_origin_color.py', '--process_id', task_list.pop(),
                                                 '--dataset_dir', '/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/real_models/dataset_v1/cast_images',
                                                 '--product_name', package_name,
                                                 '--axis', f'{axis_name}_axis',
                                                 '--img_size', '256'])
                        proc_list.append(proc)
                for proc in proc_list:
                    # print(f'state: {proc.poll()}')
                    if proc.poll() is not None:
                        proc_list.remove(proc)


if __name__ == "__main__":
    main()
