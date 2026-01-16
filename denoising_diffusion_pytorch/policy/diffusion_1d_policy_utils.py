

import torch



# def get_2d_image_to_1d(image, grid_3_dim , is_shuffle):
#         mini_batch_image    = get_2d_image_to_mini_batch_image(image, grid_3_dim, "z")
#         mini_batch_dim      = mini_batch_image.shape[0]

#         # インデックスを作成 (0から15の範囲)
#         indices = torch.tensor([[i, j, k] for i in range(mini_batch_dim) for j in range(mini_batch_dim) for k in range(mini_batch_dim)])  # shape: [4096, 3]
#         values  = mini_batch_image[indices[:, 0], indices[:, 1], indices[:, 2]]
#         result  = torch.cat((indices/(mini_batch_dim-1.0), values), dim=1)
#         # result  = torch.cat((indices, values), dim=1)
#         # result  = torch.cat((indices, values), dim=1)

#         if is_shuffle is True:
#             result_  =result[torch.randperm(result.size(0))]
#             result_tp = torch.permute(result_,(1,0))
#         else:
#             result_tp = torch.permute(result,(1,0))


#         return result_tp

# def get_2d_image_to_mini_batch_image(image=None, grid_3dim=16, permute = "z"):
#     grid_2dim    = image.shape[0]
#     grid_3dim    = grid_3dim
#     batch_img_len = int(grid_2dim/grid_3dim)

#     batch_2d_image_ = torch.zeros((grid_3dim, grid_3dim, grid_3dim, 3))
#     k = 0
#     for j in range(batch_img_len):
#         for i in range(batch_img_len):
#             batch_2d_image_[k] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
#             k = k+1

#     if permute == "z":
#         batch_2d_image  = batch_2d_image_
#     elif permute == "y":
#         batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
#     elif permute == "x":
#         batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)

#     return batch_2d_image


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