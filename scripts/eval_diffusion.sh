#!/usr/bin/bash


# python3 scripts/eval_image_diffusion_v6.py
# python3 scripts/post_process_data.py

# python3 scripts/eval_image_diffusion_v6_tmp.py 
# python3 scripts/post_process_data.py 



##############################
## for simple model
###############################
python3 scripts/eval_image_diffusion_v7_simple_model.py --config config.vae_simple_model
python3 scripts/post_process_data_v3.py --config config.vae_simple_model


# ##############################
# ## for real model
# ###############################
# python3 scripts/eval_image_diffusion_v7_simple_model.py --config config.vae
# python3 scripts/post_process_data_v3.py --config config.vae



# 計測終了
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
minutes=$(( elapsed / 60 ))
seconds=$(( elapsed % 60 ))
echo "Total execution time: ${minutes} min ${seconds} sec"
