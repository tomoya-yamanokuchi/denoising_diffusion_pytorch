import torch
import math



def get_slice_image_from_voxel_grid(image: torch.Tensor) -> torch.Tensor:
    """
    Convert voxel grid image [D, H, W, C] to tiled slice image [H2, W2, C].

    For example:
      D = 16 -> 4 x 4 tiles -> 64 x 64
      D = 49 -> 7 x 7 tiles -> 343 x 343
    """
    dim, _, _, channel = image.shape
    batch_img_len = int(math.sqrt(dim))

    if batch_img_len * batch_img_len != dim:
        raise ValueError(
            f"grid_3dim must be a perfect square for tiled image conversion. "
            f"Got dim={dim}."
        )

    image = image.view(batch_img_len, batch_img_len, *image.shape[1:])

    cast_image = image.permute(0, 2, 1, 3, 4).reshape(
        batch_img_len * image.shape[2],
        batch_img_len * image.shape[3],
        channel,
    )

    return cast_image


def get_1d_samples_to_2d_images(
    all_samples: torch.Tensor,
    grid_3dim: int,
    grid_2dim: int | None = None,
) -> torch.Tensor:
    """
    Convert diffusion 1D output [B, 6, N] to tiled 2D images [B, 3, H, W].

    all_samples:
      [B, 6, N]
      channel 0:3 = normalized xyz position in [0, 1]
      channel 3:6 = RGB value
    """
    device = all_samples.device
    grid_2dim = grid_2dim or int(math.sqrt(grid_3dim)) * grid_3dim

    all_samples_tp = torch.permute(all_samples, (0, 2, 1))  # [B, N, 6]

    all_samples_batch = torch.zeros(
        all_samples.shape[0],
        3,
        grid_2dim,
        grid_2dim,
        device=device,
        dtype=all_samples.dtype,
    )

    for i in range(all_samples_batch.shape[0]):
        all_samples_tp_index = torch.round(
            all_samples_tp[:, :, :3] * (grid_3dim - 1.0)
        ).long()[i]

        # ここが重要: 15 固定ではなく grid_3dim - 1 にする
        all_samples_tp_index = torch.clamp(
            all_samples_tp_index,
            0,
            grid_3dim - 1,
        )

        all_samples_tp_values = all_samples_tp[:, :, 3:][i]

        voxel_grid = torch.zeros(
            grid_3dim,
            grid_3dim,
            grid_3dim,
            3,
            device=device,
            dtype=all_samples.dtype,
        )

        voxel_grid[
            all_samples_tp_index[:, 0],
            all_samples_tp_index[:, 1],
            all_samples_tp_index[:, 2],
        ] = all_samples_tp_values

        slice_image = get_slice_image_from_voxel_grid(voxel_grid)
        all_samples_batch[i] = torch.permute(slice_image[None, :, :, :], (0, 3, 1, 2))[0]

    return all_samples_batch



def get_2d_image_to_1d(image, grid_3_dim , is_shuffle):
        mini_batch_image    = get_2d_image_to_mini_batch_image(image, grid_3_dim, "z")
        mini_batch_dim      = mini_batch_image.shape[0]

        indices = generate_3d_indices(mini_batch_dim=mini_batch_dim).to(image.device)
        values  = mini_batch_image[indices[:, 0], indices[:, 1], indices[:, 2]]
        result  = torch.cat((indices/(mini_batch_dim-1.0), values), dim=1)

        result_tp = torch.permute(result,(1,0))

        return result_tp


def generate_3d_indices(mini_batch_dim):
    r = torch.arange(mini_batch_dim)
    zz, yy, xx = torch.meshgrid(r, r, r, indexing='ij')  # shape: [D, D, D]
    indices = torch.stack([zz, yy, xx], dim=-1)  # shape: [D, D, D, 3]
    return indices.reshape(-1, 3)  # → [D³, 3]


def get_2d_image_to_mini_batch_image(image, grid_3dim, permute):
    # grid サイズ
    patch_size = grid_3dim

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
        import ipdb;ipdb.set_trace()
    # elif permute == "y":
    #     batch_2d_image  = patches.transpose(1,0,2,3)
    # elif permute == "x":
    #     batch_2d_image  = patches.transpose(2,1,0,3)

    return batch_2d_image
