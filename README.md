
# Install package

```git@github.com:naist-robot-learning/denoising_diffusion_pytorch.git```，ブランチ名```dev_0.1```をclone

```bash
$ pip3 install -e .
```

or

```bash
$ pip3 install denoising_diffusion_pytorch
```


---
# 📦 Dataset Preparation

## 1. 簡易形状モデルの生成

1. 簡易形状モデルのデータセットは郭先生が作成した以下のリポジトリを元に作成

   ```
   git@github.com:naist-robot-learning/nedo-dismantling-PyBlender.git
   ```

   使用するブランチは```haxhi_main```

2. リポジトリの **README** に従い，**Blender** を **Python** から呼び出せるように設定

3. 以下のコマンドを実行して、**箱の中に3つのオブジェクトがランダムに配置されたSTLデータ**を生成.
   ```bash
   blender --background --python func/obj_generator.py --object {name_of_object}
   ```
   この時点では，各オブジェクトの色は設定されない．配置を決めるだけ．後のコードでボクセル変換後，2D画像化する際に，各内部部品の属性を色情報として付与する

4. 配置パターンや生成するデータ数などは、以下のスクリプト内で設定している

   ```
   nedo-dismantling-PyBlender/func/obj_generator.py
   ```

   外形の大きさも設定しているが、**今回は外形を単一形状で統一しているため利用していない**

## 2. ボクセルへの変換と2D画像生成

1. 生成した配置データセットをボクセル表現に変換し，2D画像を生成する  
   ```git@github.com:naist-robot-learning/denoising_diffusion_pytorch.git```のリポジトにある以下のコマンドを実行

   ```bash
   python3 scripts/generate_voxel_image_w_multi_color.py --config config.vae_simple_model
   ```

2. 基本的な引数設定は、`config/` 以下にある `config.vae` などの設定ファイル（`{file_name}.py`）内で行われてる．  
   **実行ファイル内では別途詳細な設定**を行っている．

3. 主な設定項目は以下の通り：
   * **ボクセル空間のサイズ・分割数**: `grid_config`(config ファイルから読み出し)
   * **変換したいデータセットのパス**: `dataset_path`(実行ファイル内で設定)
   * **各内部部品の色指定**: `color_list`(実行ファイル内で設定)


## 3. 生成画像の整理

生成された各 2D 画像を、**1つのフォルダにまとめておく**
`mv` コマンドなどで手動で移動しました．

## 🧩 参考：現状の配置パターン設定

以下に、現状の `box_arrange_config` と各 `color` 設定の例を示します。
長いため、クリックして展開して確認してください。

<details>
<summary>クリックして展開（現状の配置パターン設定）</summary>

```python
        # dataset_4_12900k geom_test_1
        box_arrange_config = {
                "Box_1": {
                "position": { "min": [0.0, -0.02, 0.0], "max": [0.0, -0.005, 0.0] },
                "size": { "min": [0.08, 0.0, 0.05], "max": [0.1, 0.01, 0.1] }
                },
                "Box_2": {
                "position": { "min": [0.01, 0.005, 0.0], "max": [0.03, 0.02, 0.0] },
                "size": { "min": [0.03, 0.01, 0.02], "max": [0.04, 0.02, 0.05] }
                },
                "Box_3": {
                "position": { "min": [-0.01, 0.0, 0.0], "max": [-0.03, 0.02, 0.0] },
                "size": { "min": [0.01, 0.01, 0.01], "max": [0.03, 0.03, 0.03] }
                }
                }
        color_1=[0.8,0.2,0.2]
        color_2=[0.8,0.8,0.2]
        color_3=[0.2,0.8,0.8]

        # # dataset_4_12900k geom_test_2
        box_arrange_config = {
                                "Box_1": {
                                "position": { "min": [0.0, -0.02, 0.0], "max": [0.0, -0.005, 0.0] },
                                "size": { "min": [0.08, 0.0, 0.05], "max": [0.1, 0.01, 0.1] }
                                },
                                "Box_2": {
                                "position": { "min": [0.01, 0.005, 0.0], "max": [0.03, 0.02, 0.0] },
                                "size": { "min": [0.03, 0.01, 0.02], "max": [0.04, 0.02, 0.05] }
                                },
                                "Box_3": {
                                "position": { "min": [-0.01, 0.0, 0.0], "max": [-0.03, 0.02, 0.0] },
                                "size": { "min": [0.01, 0.01, 0.01], "max": [0.03, 0.03, 0.03] }
                                }
                                }
        # geom_test_2
        color_1=[0.9,0.2,0.2]
        color_2=[0.2,0.8,0.8]
        color_3=[0.8,0.8,0.2]


        # # # dataset_4_12900k geom_test_3
        box_arrange_config ={
                                "Box_1": {
                                "position": { "min": [0.0, -0.02, 0.0], "max": [0.0, -0.005, 0.0] },
                                "size": { "min": [0.08, 0.0, 0.05], "max": [0.1, 0.01, 0.1] }
                                },
                                "Box_2": {
                                "position": { "min": [0.01, 0.005, 0.0], "max": [0.03, 0.02, 0.0] },
                                "size": { "min": [0.03, 0.01, 0.02], "max": [0.04, 0.02, 0.05] }
                                },
                                "Box_3": {
                                "position": { "min": [-0.01, 0.0, 0.0], "max": [-0.03, 0.02, 0.0] },
                                "size": { "min": [0.01, 0.01, 0.01], "max": [0.03, 0.03, 0.03] }
                                }
                                }
        # # geom_test_3
        color_1=[0.2,0.8,0.8]
        color_2=[0.9,0.2,0.2]
        color_3=[0.8,0.8,0.2]


        dataset_5_12900k geom_test_1
        box_arrange_config = {
                                "Box_1": {
                                "position": { "min": [0.0, -0.02, 0.0], "max": [0.0, -0.005, 0.0] },
                                "size": { "min": [0.08, 0.0, 0.05], "max": [0.1, 0.01, 0.1] }
                                },
                                "Box_2": {
                                "position": { "min": [-0.01, 0.0, 0.0], "max": [-0.03, 0.02, 0.0] },
                                "size": { "min": [0.03, 0.01, 0.02], "max": [0.04, 0.02, 0.05] }
                                },
                                "Box_3": {
                                "position": { "min": [0.01, 0.005, 0.0], "max": [0.03, 0.02, 0.0] },
                                "size": { "min": [0.01, 0.01, 0.01], "max": [0.03, 0.03, 0.03] }
                                }
                                }
        # geom_test_1
        color_1=[0.9,0.2,0.2]
        color_2=[0.8,0.8,0.2]
        color_3=[0.2,0.8,0.8]
        
        # dataset_6_13900k geom_test_1=
        box_arrange_config = {
                                "Box_1": {
                                "position": { "min": [0.0, -0.04, 0.0], "max": [0.0, -0.0, 0.0] },
                                "size":     { "min": [0.08, 0.0, 0.05], "max": [0.1, 0.015, 0.1] }
                                },
                                "Box_2": {
                                "position": { "min": [-0.01, 0.0, 0.0], "max": [-0.03, 0.02, 0.0] },
                                "size":     { "min": [0.03, 0.01, 0.02], "max": [0.04, 0.02, 0.05] }
                                },
                                "Box_3": {
                                "position": { "min": [0.01, 0.005, 0.0], "max": [0.03, 0.02, 0.0] },
                                "size":     { "min": [0.01, 0.01, 0.01], "max": [0.03, 0.03, 0.03] }
                                }
                                }
        # geom_test_1
        color_1=[0.8,0.8,0.2]
        color_2=[0.2,0.8,0.8]
        color_3=[0.9,0.2,0.2]
```

</details>


---

# 🚀 Train conditional diffusion model

以下の学習コードを実行する
```
python3 scripts/train_cond_image_diffusion_v1.py --config config.vae_simple_model
```
主要な学習設定は```config/vae.py```内で設定．（時々プログラム内でハードコーディングされてるときもあります．．．）

<details>
<summary>configの大まかな設定内容（クリックして展開）</summary>

```python

'conditional_image_diffusion': {
    'USER_NAME'         : "user", #"haxhi", #rootとなるuser名

    ## model
    'model'             : 'models.unet_2d_simple_devel2.Unet', # 拡散モデルで利用するUnet"denoising_diffusion_pytorch/denoising_diffusion_pytorch/models/unet_2d_simple_devel2.py"内のUnetクラスを指定して読み込むことになる

    ## モデルの設定
    'dim_mults'         : (1, 2, 4, 8),
    'flash_attn'        : True,
    'self_condition'    : False, # default = False
    'init_dim'          : 64,

    'diffusion'         : 'models.conditional_image_diffusion_cfg_devel2.GaussianDiffusion', #拡散モデルのクラス
    'beta_schedule'     : 'sigmoid', # default = 'sigmoid' ['sigmoid', 'cosine']
    'n_diffusion_step'  : 1000, # default = 1000
    'sampling_step'     : 20,

    ## dataset
    "loader":"data_loader.cond_image_data_loader.Cond_image_dataloader",
    "dataset_config": {"dataset" : {"name": "celeba",
                                    "path": "/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_4142435161_13901k/voxel_images_w_multi_color_v1",
                                    "min": -1,
                                    "max": 1,
                                    "h": 32, # only effects 'type=None or center'
                                    "type": "pattern",
                                    "p": 0.2} # only effects 'type=random'
                        },
    'image_size' : 64,


    ## serialization
    'logbase'           : 'logs',
    'prefix'            : 'conditional_diffusion2/',
    'tag'               : "flower_image_v1",
    'exp_name'          : watch(conditional_image_diffusion_train_args_to_watch),

    ## training
    'trainer'           : 'trainer.diffusion_conditional_image_trainer.Trainer',
    'batch_size'        : 96, # default
    'learning_rate'     : 8e-5, # default
    'train_step'            : 800000,           # total training steps
    'save_and_sample_every' : 2000,
    'gradient_accumulate_every' : 2,        # gradient accumulation steps
    'ema_decay'          : 0.995,           # exponential moving average decay
    'amp'                : True,            # turn on mixed precision
    'calculate_fid'      : False,
    'device'             :'cuda:0'
},

```

</details>

```logs/```以下に学習結果が保存される．学習ロスや学習途中で推論した結果などはtensorflowで見れる．

---

# 📊 Evaluation with cutting env
以下のコードを実行して，学習した内部構造推論モデルに基づいた切断計画による内部部品抽出タスクを行う．
```bash
python3 scripts/eval_image_diffusion_v7_simple_model.py --config config.vae_simple_model
```

主要な学習設定は```config/vae.py```内で設定．（時々プログラム内でハードコーディングされてるときもあります．．．）

<details>
<summary>configの大まかな設定内容（クリックして展開）</summary>

```python
    'diffusion_plan': {
        'USER_NAME'         : "user",

        'policy'            :'policy.cutting_surface_planner_v9.cutting_surface_planner', # for simple model 切断計画で使うpolicy
        'batch_size'        : 32,  # diffusion default 内部構造推論で一度に推論するサンプル数


        ## policy_config
        'policy_config'     :{
                                'ctrl_mode':"epsilon_greedy_00", # 提案法
                                # 'ctrl_mode':"prior_based_ep_00",# temp
                                # 'ctrl_mode':"random",
                                # 'ctrl_mode':"no_cond",
                                # 'ctrl_mode':"oracle_obs",
                                #################################################################################################################
                                ## 内部部品の種類を判定するマスク値の設定（簡易モデルと複雑モデルで学習時の各モデルの色設定が違うので，ここで変えないといけない．．．）
                                ##################################################################################################################
                                "image_mask_config_b" : {"target_mask"  :np.asarray([0.2,0.8,0.8]),
                                                        "target_mask_lb":np.asarray([0.2,0.8,0.8])-np.asarray([0.25,0.25,0.25]),
                                                        "target_mask_ub":np.asarray([0.2,0.8,0.8])+np.asarray([0.25,0.25,0.25])},
                                "image_mask_config_r": {"target_mask"   :np.asarray([0.8,0.2,0.2]),
                                                        "target_mask_lb":np.asarray([0.8,0.2,0.2])-np.asarray([0.1,0.1,0.1]),
                                                        "target_mask_ub":np.asarray([0.8,0.2,0.2])+np.asarray([0.2,0.2,0.2])},
                                "image_mask_config_y": {"target_mask"   :np.asarray([0.8,0.8,0.2]),
                                                        "target_mask_lb":np.asarray([0.8,0.8,0.2])-np.asarray([0.1,0.1,0.1]),
                                                        "target_mask_ub":np.asarray([0.8,0.8,0.2])+np.asarray([0.2,0.2,0.6])},
                                # "image_mask_config_b" : {"target_mask"  :np.asarray([0.0,0.0,1.0]),
                                #                         "target_mask_lb":np.asarray([0.0,0.0,1.0])-np.asarray([0.1,0.1,0.1]),
                                #                         "target_mask_ub":np.asarray([0.0,0.0,1.0])+np.asarray([0.1,0.1,0.0])},
                                # "image_mask_config_r": {"target_mask"   :np.asarray([1.0,0.0,0.0]),
                                #                         "target_mask_lb":np.asarray([1.0,0.0,0.0])-np.asarray([0.1,0.1,0.1]),
                                #                         "target_mask_ub":np.asarray([1.0,0.0,0.0])+np.asarray([0.0,0.1,0.1])},
                                # "image_mask_config_y": {"target_mask"   :np.asarray([0.0,1.0,0.0]),
                                #                         "target_mask_lb":np.asarray([0.0,1.0,0.0])-np.asarray([0.1,0.1,0.1]),
                                #                         "target_mask_ub":np.asarray([0.0,1.0,0.0])+np.asarray([0.1,0.0,0.1])},
                                ##############################################################################################################
                                ## 推論モデルの設定，このタグ自体の名前がdiffusion_planになってるけど他の比較手法でもこのtagを使う．そのためここで別途
                                ## 推論モデルを設定する必要がある．．．（このタグ自体の名前がdiffusion_planがイケていないです．．）
                                ##############################################################################################################
                                # 'infer_model':"vaeac",
                                # 'infer_model':"diffusion",
                                'infer_model':"conditional_diffusion",
                                # 'infer_model':"diffusion_1D",
                                ##############################################################################################################
                                ## 切断リスクの許容値，論文内での/eta
                                ##############################################################################################################
                                "decision_mode": {
                                                    "mode": "clip_ucb_raw",
                                                    "param":{
                                                            "ucb_lb":0.5,  # 1.0 or 0.99 or
                                                             }
                                                    },
                                ###############################################################################################################
                                ## 拡散モデルで推論する際のガイダンススケール 論文内のプレリミで出てきた\omega
                                ###############################################################################################################
                                "cfg_omega": 0.2,
                                },
        ## eval data loading
        'eval_data_path'    : 'f:/home/{USER_NAME}/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1',
        'eval_data_lists'    : {
                                'Object_1':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_3/Boxy_0",
                                'Object_2':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_3/Boxy_1",
                                'Object_3':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_3/Boxy_2",
                                'Object_4':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1/Boxy_0",
                                'Object_5':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1/Boxy_1",
                                'Object_6':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1/Boxy_2",
                                'Object_7':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1/Boxy_0",
                                'Object_8':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1/Boxy_1",
                                'Object_9':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1/Boxy_2",
                                },



        ##############################
        ## simple model prefix
        ###############################
        'diffusion_loadpath': 'f:/home/{USER_NAME}/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/conditional_diffusion2/T1000_D64_dataset_13901k_v11',
        'diffusion_epoch'   : '100000',
        'prefix'            : 'diffusion_plans/dataset_4142435161_13901k_v1_2/',


        'task_step' :8,

        ######################################################
        ##初期切断範囲の設定．rangeの一番最後の切断位置が観測される
        ######################################################
        "start_action_idx":{
                            "obj_tag":"123456789",
                            "Object_1":np.arange(47,41-1,-1), #41
                            "Object_2":np.arange(47,41-1,-1), #41
                            "Object_3":np.arange(47,41-1,-1), #41
                            "Object_4":np.arange(0,4+1),   #4
                            "Object_5":np.arange(0,4+1),   #4
                            "Object_6":np.arange(0,4+1),   #4
                            "Object_7":np.arange(0,4+1),   #4
                            "Object_8":np.arange(0,4+1),   #4
                            "Object_9":np.arange(0,4+1),   #4
                            },         


        ## serialization
        'logbase'           : 'logs',
        'suffix'            : 'f:{policy_config["ctrl_mode"]}',
        'observation_mode'  : 'partial_obs',
        'iter'              : [0,6],
        'tag'               : 'f:B{batch_size}_T{task_step}_{observation_mode}_{policy_config["infer_model"]}_a{start_action_idx["obj_tag"]}_{policy_config["decision_mode"]["mode"]}_{policy_config["decision_mode"]["param"]["ucb_lb"]}_v12_1_for_paper_render',
        'exp_name'          : watch(diffusion_plan_args_to_watch),
        'device': 'cuda:0',
    },
```
</details>

## ⚠️ 切断環境のenvコード内部```denoising_diffusion_pytorch/denoising_diffusion_pytorch/env/voxel_cut_sim_v1.py```でも，真の内部部品の体積値等を計算するために内部部品の種類を判定するマスク値を設定する箇所があります．

現状，簡易モデルと複雑形状モデルでマスク値の設定が違うので状況に応じてコメントアウトで対応する必要があります．


----
# 切断環境の説明

一通り動作確認ができたら．．．．

# 切断アルゴリズムの説明

一通り動作確認ができたら．．．．




# How to create sphinx docs

Install dependencies

```
pip3 install sphinx sphinx-rtd-theme sphinx_fontawesome
pip3 install myst_parser
```

create docs folder
```
mkdir docs
```
多分docsの中で実行する．
```
python3 -m sphinx.cmd.quickstart
```
```
mkdir docs/source/resources
```
docs のdirectoryの一個上で実施する
```
<!-- sphinx-apidoc --force -o docs/source/resources denoising_diffusion_pytorch/env   -->
sphinx-apidoc --force -o docs/source/resources  src/my_module  --implicit-namespaces
```
```
python3 -m sphinx.cmd.build -a -b html docs/source docs/build
```





---
# 以下は関係ありません
## Demo code
```bash
$ python3 scripts/native/simple_diffusion.py
```


## latest usage

- train
    ```
    python3 scripts/train_image_diffusion_v2.py --config config.vae --dataset Image_diffusion_2D
    ```

- test
    ```
    python3 scripts/eval_image_diffusion_v2.py    --config config.vae --dataset Image_diffusion_2D
    ```
- eval test results
    ```
    python3 scripts/diffusion_post_process.py    --config config.vae --dataset Image_diffusion_2D
    ```

- render denoising process
    ```
    python3 scripts/render_denoising_process.py    --config config.vae --dataset Image_diffusion_2D
    ```


---

## conventional usage

- train and test
    ```
    python3 scripts/native/simple_diffusion_v1.py
    ```

- eval test results
    ```
    python3 scripts/native/post_process.py
    ```

- render denoising process
    '''
    python3 scripts/native/render_denoising_process.py
    '''


<!-- --- -->

```
create docs folder
mkdir docs
多分docsの中で実行する．
python3 -m sphinx.cmd.quickstart
mkdir my_docs/source/resources
docs のdirectoryの一個上で実施する
sphinx-apidoc --force -o docs/source/resources denoising_diffusion_pytorch/env
python3 -m sphinx.cmd.build -b  -a html docs/source docs/build
```



