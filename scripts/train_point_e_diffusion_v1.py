


import os
import torch


from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.config import Config

torch.cuda.empty_cache()
torch.cuda.ipc_collect()


class Parser(parser):
    dataset: str = 'Image_diffusion_2D'
    config: str =  'config.vae'

#---------------------------------- setup ----------------------------------#


args = Parser().parse_args('point_e_diffusion')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = Config(
    args.loader,
    savepath   = (args.savepath, 'dataset_config.pkl'),
    folder     = args.dataset_path,
    image_size = args.image_size,
    grid_3dim  = args.grid_3dim,
    is_shuffle = args.is_shuffle,
    augment_horizontal_flip = args.horizontal_flip,
    convert_image_to = args.convert_image_to,
)



dataset = dataset_config()
image_size        = dataset.image_size
channels, seq_len = dataset.__getitem__(1).shape


# import ipdb;ipdb.set_trace()

model_config = Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    device_name=args.device,
    dtype=torch.float32,
    input_channels=channels,              # 入力次元：点の特徴
    output_channels=channels,             # 出力も同じ（εの予測）
    n_ctx=args.n_ctx,                        # 1サンプルあたりのトークン数（点数や系列長）
    width=args.width,                     # Transformer 内のトークン埋め込み（特徴ベクトル）の次元数
    layers=args.layers,                   # Transformer ブロックの層数（Self-Attention + FFN のセット）深くするほど高次特徴が抽出可能
    heads=args.heads,                     # Multi-Head Attention のヘッド数（注意の並列分割） 各ヘッドは width/heads = 64 次元）
    time_token_cond=args.time_token_cond  # t を "special token" としてトークン列に追加するかどうか t 埋め込みをトークンとして先頭に追加（Transformer で直接処理される）
)


diffusion_config = Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    seq_length = seq_len,
    timesteps = args.n_diffusion_step,
    sampling_timesteps = args.sampling_step, # number of steps
    beta_schedule = args.beta_schedule,      # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    device = args.device
)





trainer_config  = Config(
    args.trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size = args.batch_size,
    train_lr         = args.learning_rate,
    train_num_steps  = args.train_step,         # total training steps
    save_and_sample_every  = args.save_and_sample_every, 
    gradient_accumulate_every = args.gradient_accumulate_every,    # gradient accumulation steps
    ema_decay                 = args.ema_decay,                # exponential moving average decay
    amp                       = args.amp,                       # turn on mixed precision
    results_folder            = args.savepath
)




model       = model_config()
diffusion   = diffusion_config(model)
trainer     = trainer_config(diffusion_model = diffusion, dataset = dataset)


original_config_path = args.savepath
original_config_path = os.path.join(original_config_path,"original_configs_backup.py")
args.save_config_file(original_config_path)

print(args.savepath)


trainer.train()

# import ipdb;ipdb.set_trace()