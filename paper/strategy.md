# Revision Strategy — APIN-D-26-02746

針對 `paper/comments.md` 中的 **AE** 與 **Reviewer #3** 意見，依目前 repo 結構提出具體執行策略。
（Reviewer #1 已被 AE 判定為與本論文無關，不在 scope 內。）

每一條包含：

- **C** (Concern)：reviewer 關切點
- **S** (Current state)：repo 現況與相關檔案
- **P** (Plan)：建議的執行步驟與要修改 / 新增的檔案
- **D** (Deliverable)：論文 revision 中對應的圖、表、文字修改

> 注意：本檔為作者間共識文件（R1–R6 debate 結束後鎖定）。所有 forward-deployment 語言已剔除；evaluation venue 為 simulation by deliberate design，論證見 §3 W4。

---

## 0.5. Sim-only sufficiency argument — load-bearing thesis

**Constraint**：本 revision 中**無實體實驗**。Evaluation venue 為 simulation。

**Load-bearing answer**：論文的 contributions 是 **planning algorithm + analysis + design abstraction**（不是 a physical system）。在此 contribution scope 下，simulation 是 the appropriate evaluation venue。這個論點寫在新的 Discussion 子節 **W4「Scope of Evaluation: Why Simulation is the Appropriate Venue」**（§3 W4），以 four-anchor scaffold 正面論證，禁用任何退讓性 forward-deployment 語言（具體禁用 pattern 列表見 §3 W1）。（**Note**：W2 包含一條 measured 的 real-robot extension 句，這與本 thesis 不衝突——禁用的是把本投稿 frame 為 preliminary / stepping-stone 的措辭，**不**禁用 W2 風格的「real-robot 整合是 natural extension，且 not required for the contributions claimed」這類 scope-clarifying 句。）所有與 AE-4 / R3-1 / R3-9 相關的回應，最終都回扣到 W4。

所有 reviewer/AE 對「real environment / deployment feasibility」的關切，在本 revision 中以下列三路答覆：
1. **Specification surface**（§3 W5）：把假設、雜訊預算、安全 margin、運算 envelope 以表格形式公開，供 integrator 組合自己的 perception / cutting 模組。
2. **`feature_provider` 介面 + 三種 instantiations**（R3-9）：把 perception 顯式從 planner contribution 切出，並以三個 provider 證明介面非空話。
3. **Claim moderation**（W1）：Abstract / Introduction / Conclusion 的 verb 全面改寫，禁止任何隱含實機結果的 verb。

---

## 1. AE 高階要求 → 模組對應

| AE 要求 | 主要對應 repo 模組 | 對應到 R3 子點 |
|---|---|---|
| AE-1 參數合理性（cutting-risk threshold、guidance scale、sampling、voxel resolution） | `config/eval/policy/default.yaml`、`config/inferencer/conditional_diffusion.yaml`、`denoising_diffusion_pytorch/policy/decision/decision_rules.py`、`models/proposed/conditional_diffusion_cfg.py` | R3-2, R3-5, R3-8, R3-10 |
| AE-2 iterative loop 收斂性 | `denoising_diffusion_pytorch/policy/cutting_surface_planner_v9.py`、`denoising_diffusion_pytorch/eval/episode_runner.py` | R3-3 |
| AE-3 baseline 公平性 | `denoising_diffusion_pytorch/models/baselines/vaeac/`、`trainer/vaeac_trainer.py`、`config/inferencer/vaeac.yaml` | R3-6 |
| AE-4 scalability / "deployment feasibility" → **重新詮釋為 operating-envelope characterization**（見 §3 W4 / W5、§7） | `models/experimental/{dit,uvit}.py`、`env/voxel_cut_sim_v1.py`、`cost/color_mask_cost_estimator.py`、新 `perception/` | R3-1, R3-7, R3-8, R3-9 |
| AE-5 moderate claims | Abstract / Introduction / Conclusion / Limitation | 全體 |

→ revision 拆成 **四批實驗 (E1–E4)** + **五段文字修改 (W1–W5)**。

---

## 2. Reviewer #3 — 逐點執行策略

### R3-1 ｜ 真實環境雜訊與切割誤差未納入

- **C**：framework 在 noise-free observation 與 ideal cutting 下驗證，未說明對 sensing noise / cutting-pose error 的健壯性。
- **S**：
  - `env/voxel_cut_sim_v1.py` 由 `voxel_cut_handler` 直接做 axis-aligned 切割，沒有雜訊注入機制。
  - 觀測來自 `cast_2d_image_to_box_color`（`cost/color_mask_cost_estimator.py`），亦為理想色彩讀取。
- **P**（**E1: noise & perturbation ablation**）：
  1. 在 **mask domain**（policy 真正消費的訊號）注入觀測雜訊，而非僅 upstream RGB jitter：
     - random binary flip：`p_flip ∈ {0.0, 0.01, 0.05, 0.1}`
     - morphological dilation / erosion：`r ∈ {0, 1, 2, 3}` voxels
     - false-positive blobs：每張 mask 0–3 個隨機 blob，半徑 1–2 voxels
     - 入口：將 corruption 包裝為 `MaskCorruptionProvider`（見 R3-9，feature_provider 介面的其中一個 instantiation）。
  2. 切割執行端僅加 **translational plane perturbation**（沿法向位移 `Δ ∈ {0, ±1, ±2} voxels`）。
     - **明確排除角度抖動 θ**：non-axis-aligned cuts 屬於本論文未建模的 cutting-action class，重構 `voxel_cut_handler` / `pv_box_array_multi_type_obj` 為非軸對齊需數週工程量，超出本 revision scope。Limitation (W2) 明寫「non-axis-aligned cuts are outside the cutting-action class modeled in this paper」。
  3. 透過 Hydra config 新增 `config/env/noisy.yaml` 切換；`episode_runner` 不需改邏輯。
  4. 固定 **3 個代表性物件** × 雜訊網格，計算 task success / over-cut / damage rate。
- **D**：
  - 表：robustness ablation（success rate vs mask-corruption level；plane-translation Δ 為獨立列）。
  - 圖：success-rate 對 mask-corruption 強度的曲線。
  - 文字：Limitation (W2) 對適用雜訊上限明示，並包含 cutting-action class scope 的明文。

### R3-2 ｜ 參數調整策略缺乏依據

- **C**：超參數選擇沒有 justification，diffusion 行為在不同任務下無法解釋。
- **S**：
  - 主要超參數集中於 `config/inferencer/conditional_diffusion.yaml`（`beta_schedule=sigmoid`、`timesteps=1000`、`sampling_timesteps=20`、`network.dim=64` 等）與 `config/eval/policy/default.yaml`（`guidance_scale`、`ucb_lb`、`sample_image_num=32`）。
  - 目前論文僅報告一組「best config」，未附 sweep。
- **P**（**E2-a: hyperparameter sensitivity sweep — 單一 sweep 矩陣**）：
  1. 將 R3-2 / R3-4 / R3-5 / R3-8 / R3-10 的 sweep 統合為**單一 E2 sweep 矩陣**，共用同一組固定 evaluation episode set，避免重複跑 startup。
  2. 每個 swept parameter **預先（看 success 之前）宣告一個 named selection criterion**（見 §3 W3 / 下表）。
  3. 透過 `scripts/train/run_train.py` 與 `scripts/eval/run_eval.py` 的 Hydra override 撰寫 sweep batch script 於 `scripts/eval/sweeps/`。
  4. 每組以固定 seed × ≥3 episode 重複。

**Pre-declared selection-criterion table** (W3 引用)：

| Parameter | Selection criterion (pre-declared, before consulting success rate) | Sweep grid |
|---|---|---|
| ω (CFG guidance scale) | Brier-score minimum on dev set, sliced by observation regime | {0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0} |
| `ucb_lb` (cutting-risk threshold) | Pareto knee on damage-vs-removal curve, dev split | {0.1, 0.3, 0.5, 0.7, 0.9, 0.99} |
| `ucb_beta` | Variance-of-decisions stability threshold | {0, 0.5, 1, 2, 3} |
| `sampling_timesteps` (DDIM steps) | Latency budget at chosen resolution | {5, 10, 20, 50, 100, 250} |
| Voxel resolution | Boundary-IoU ≥ τ at minimum FLOPs | simple: {16, 25, 36, 49, 64}; real: {128, 256, 344} |
| `task_step` (planning horizon) | Marginal success gain < 1% per added step | {4, 6, 8, 12} |

- **D**：
  - 表：上方 selection-criterion table（W3 中以正式表呈現）。
  - 表：主 sweep 統一表（rows = 參數值；cols = task success / Brier / latency / 安全指標）。
  - 文字：Method (W3) 對每個參數的選取依據與 trade-off 寫一段話，**並明引上述 selection criterion**。

### R3-3 ｜ Iterative estimation 缺乏收斂性說明

- **C**：iterative estimation–planning loop 是否收斂、停止條件為何、序列更新如何影響 planning 可靠性，皆未說明。
- **S**：
  - `cutting_surface_planner_v9.py` 與 `eval/episode_runner.py` 是 loop 主體。`task_step: 8`（`config/eval/default.yaml`）為固定 horizon，沒有 early-stop。
  - 每步輸出 ensemble presence score（`policy/ensemble_image_builder.py`）與 decision aggregator（`policy/decision/decision_aggregator.py`）。
- **P**（**E2-b: convergence diagnostics + held-out stopping-rule deployment**）：
  1. 在 `episode_runner.py` 增加 per-step logging：
     - `presence_score` 的 L1/L2 變化，**只在 uncut sub-volume 上計算**（避免 trivial tautology：cut 走的部分本來就不會再變）。
     - ensemble 預測的 voxel-wise variance / entropy（限定 uncut sub-volume）。
     - decision-rule 的 `clip_ucb_raw` UCB margin 變化。
  2. 將指標寫入 `eval/episode_artifact_manager.py` 既有的 artifact JSON。
  3. 評估 episode set 切 **30/70 dev/test split**：
     - 在 dev split 上 calibrate 停止規則「presence-score Δ (on uncut sub-volume) < ε 連續 K step」。
     - 在 test split 上 **實際 apply 該停止規則**，與固定 `task_step=8` 比較最終 success 與平均步數。
  4. 加 horizon ablation：固定 `task_step ∈ {4, 6, 8, 12}` 比較最終成功率。
  5. **Framing 明確為 empirical convergence**，不提供 theoretical guarantee，不留「如理論難給則退讓為 empirical」這種對沖句。
- **D**：
  - 圖：典型 episode 的收斂曲線（uncut-subvolume presence-score Δ、entropy、UCB margin）。
  - 表：dev-calibrated stopping rule 在 test split 的 success / 平均步數 vs 固定 horizon。
  - 表：不同 horizon 的 trade-off。
  - 文字：Method 加 stopping criterion 段，Limitation (W2) 明寫「convergence is established empirically; no theoretical guarantee is claimed」。

### R3-4 ｜ Voxelization 對精細結構表示有限

- **C**：voxel + part attribute 對細小結構過於粗糙，內部邊界精度不足。
- **S**：
  - `image_size` 在 simple model = 64×64、real model = 344×344。
  - voxel grid side length 在 `policy_config.voxel_grid_side_length`、`grid_config["side_length"]` 控制（`env/voxel_cut_sim_v1.py`）。
  - DiT/U-ViT 在 `models/experimental/` 已備好作為高解析度的替代 backbone。
- **P**（**E2-c: voxel resolution ablation**）：
  1. simple-object pipeline：grid side ∈ {16, 25, 36, 49, 64}（保留 floor end 以顯示 fine-structure 失敗點）。
  2. real-object pipeline 上限封在 **{128, 256, 344}**（512×512 retrain 在 3090 上 ~10+ days，超出 revision window；如有富餘可只跑 inference-only profile）。
  3. 報 **boundary IoU** + **surface F-score @ τ voxels**（取代 Chamfer，後者在 voxel grid 上定義不清）。
- **D**：
  - 圖：boundary IoU / surface F-score vs resolution。
  - 表：resolution × success / IoU / F-score / latency。
  - 文字：說明「voxel resolution 為 accuracy-compute trade-off」，並指明本論文用的 setting 落點（W3 引用 selection criterion）。

### R3-5 ｜ Classifier-Free Guidance 系統評估缺失

- **C**：guidance scale 對不同觀測情境的影響沒有系統評估。
- **S**：
  - CFG 實作於 `models/proposed/conditional_diffusion_cfg.py:300 / :325`：`img = (1+ω)·x_cond − ω·x_uncond`。
  - 預設 `guidance_scale: 0.2`（`config/eval/policy/default.yaml`、`config/vae.py` 內亦寫死 `cfg_omega=0.2`）。
- **P**（**E2-d: CFG sweep**）：
  1. ω ∈ {0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0} sweep。
  2. 對不同 **觀測比例**（早 / 中 / 晚 step regime）切片分析 → 直接回應 reviewer 的「different observation scenarios」。
  3. 報三類指標：
     - presence-score 對 ground truth 的 **Brier score**（取代 reliability diagram；voxel binary 上 Brier 定義清楚）。
     - sample diversity（M=32 sample 之間）。
     - 下游 task success / over-cut。
- **D**：
  - 圖：success vs ω 折線圖，分早 / 中 / 晚 observation regime 多條曲線。
  - 表：建議的 ω 預設值與 Brier-minimum 依據。
  - 文字：Method (W3) 中加「ω 選取理據」，並明引 selection criterion。

### R3-6 ｜ Baseline 設定不一致

- **C**：VAEAC 等 baseline 設定未對齊，比較不公平。
- **S**：
  - VAEAC 在 `models/baselines/vaeac/vaeac.py`、trainer 在 `trainer/vaeac_trainer.py`、config 在 `config/inferencer/vaeac.yaml`（n_hidden=32、fc_hidden=100、lr=5e-5、800k step、batch=96）。
  - Proposed CFG diffusion：`network.dim=64, dim_mults=[1,2,4,8]`、lr=8e-5、800k step、batch=96。
  - 表面上 step 與 batch 對齊，但 **參數量、optimizer schedule、capacity** 並未驗證對等。
- **P**（**E3: audit-all + matched-compute VAEAC rerun**）：
  1. 寫 `scripts/eval/baseline_audit.py`：自動列印每個 baseline 與 proposed 的 (a) 參數量、(b) FLOPs/step、(c) wall-clock 總訓練成本、(d) inference latency、(e) 資料量。
  2. **僅執行 B-equal-compute** rerun，且僅針對 **VAEAC**（reviewer 點名的 baseline）。三套對等策略全跑（VAEAC + PCD-DM + CVAE 各 ×3 = 9 retrains）超出 revision window。
  3. PCD-DM / CVAE 只在 audit 表中報差距；若差距 > 2× compute 才另行調整，否則維持原訓練結果並透明標註。
- **D**：
  - 表：baseline 對齊規格表（全 baseline 一張）。
  - 表：matched-compute VAEAC vs proposed 的 success / IoU 比較。
  - 文字：Experimental Setup 補「Fair comparison protocol」段落，明寫 scoping choice。

### R3-7 ｜ 合成資料分布過窄

- **C**：synthetic structural distribution 過小，限制對未見產品配置的泛化能力。
- **S**：
  - 資料來源：`scripts/data_generation/generate_voxel_image_w_multi_color.py`（simple）與 `..._real_obj.py`（real，含 Boxy_0/1/2 之類）。
  - 目前訓練 / 評估的 part configuration 變異受限。
- **P**（**E4-a: geometry-OOD 評估**）：
  1. 引入 **外部 mesh corpus**：PartNet / Thingi10K 子集，搭配 deterministic semantic-part → 固定 color 的 assignment script。
     - 新檔：`scripts/data_generation/ood_external_mesh.py`。
     - Color assignment 與 training-time color semantics **一致**（如 target part 永遠取 target hue），維持 conditioning 訊號有效。
  2. 切兩種 split：
     - **In-distribution test**：與訓練同 generator。
     - **OOD test**：外部 mesh + 一致色彩語義（**caption 明寫 "geometry-OOD: external mesh corpus with deterministic part-color assignment inherited from training-time color semantics"**）。
  3. **額外 row：color-permutation OOD**：隨機重排 color↔semantic-part mapping，量化 planner 對 color identity 的依賴 → 此 row 直接 justifies `feature_provider` abstraction（R3-9）。
  4. simple-pipeline 端也擴充 generator（part 數 / 大小 / 位置 / 色彩配置隨機化）。
- **D**：
  - 表：in-dist / geometry-OOD / color-permuted OOD 三 row 比較。
  - 圖：OOD 失敗 case qualitative panel。
  - 文字：Limitation (W2) 明寫「OOD here = geometry-OOD; cross-category transfer under different color semantics characterized via color-permutation row」。

### R3-8 ｜ DDIM sampling 可擴展性未探討

- **C**：DDIM 在高解析度 voxel grid 的計算成本與 real-time 可行性沒有資料。
- **S**：
  - DDIM 邏輯在 `models/proposed/conditional_diffusion_cfg.py:ddim_sample`（line 261），`sampling_timesteps` 由 config 控制（目前 20）。
  - README 宣稱 DiT 在 344×344 比 UNet 快 4.5×；此宣稱**必須由 profile 結果重現**，否則於文字中撤回。
- **P**（**E4-b: DDIM scalability profile**）：
  1. 寫 `scripts/eval/profile_sampling.py`：
     - resolution × sampling_steps × backbone (UNet vs DiT vs U-ViT) 三維 grid。
     - 紀錄 wall-clock latency、peak GPU memory、**energy = latency × measured GPU power (nvidia-smi sampling)**，不用 estimate。
  2. sampling_steps ∈ {5, 10, 20, 50, 100, 250}。
  3. 對 batch=1 與 batch=32 兩種 setting 都報。
  4. 主文呈現「品質-延遲」Pareto 圖：x = latency，y = success / Brier。
- **D**：
  - 圖：Pareto frontier（speed vs quality）。
  - 表：resolution × backbone × latency / memory / energy。
  - 文字：定位為「operating envelope characterization」，回扣 W4 / W5。

### R3-9 ｜ 特徵編碼僅用簡化 color attribute — `feature_provider` 為主文 design contribution

- **C**：voxel feature 只用簡化 color，忽略真實感測複雜度。
- **S**：
  - color mask 流程在 `cost/color_mask_cost_estimator.py`（`color_range_mask`），target color 在 `config/eval/policy/default.yaml`（blue/red/yellow segmentation）。
- **P**（**`feature_provider` 介面 + 三 instantiations，全部主文**）：
  1. 在 `denoising_diffusion_pytorch/perception/` 新增 `feature_provider.py`（abstract base）。
  2. **三個 provider**（全部主文，共用同一介面，使「介面」非空話）：
     - `ColorRangeProvider`：現行 color_range_mask 流程，retroactive 標為 baseline instantiation。
     - `MaskCorruptionProvider`：包裝 E1 的 mask-domain noise injection。
     - `KMeansRenderProvider`：
       - PyVista textured render → K-means / GMM classifier on rendered slice → mask → planner（planner / diffusion 模型**不重訓**）。
       - K-means **在 held-out render set 上 fit**（`renders_fit/`），planning 時用 `renders_eval/`，兩個 set 由 disjoint episode seeds 生成。
       - Render conditions：`lighting_neutral` + `lighting_harsh`（specular bleed）；`texture_overlay` 為 nice-to-have，若 Week 4 schedule slip 則 **drop first**。
       - 新檔：`scripts/data_generation/render_for_perception_test.py`。
  3. Method 章節增「Feature provider abstraction」一節，附介面圖與三 provider 列舉。
  4. 結果表加 `perception` column，跨切結果（同一 planner 在三 provider 下的數字）。
- **D**：
  - 表：robustness 表 + perception column（ColorRange / MaskCorruption / KMeansRender 各跑）。
  - 圖：`feature_provider` 介面示意圖（一張）。
  - 文字：
    - Method 新節「Feature provider abstraction」。
    - Discussion (W4) 用 Argument 2「separation of concerns」回扣本介面。
    - Limitation (W2) 用 scope-not-future-work wording（見 W2 bullet 3 重寫）。

### R3-10 ｜ Cutting action planning 評估含糊，安全因素未檢視

- **C**：cutting-risk threshold 設定與安全關鍵指標評估方式不明確。
- **S**：
  - cutting-risk threshold = `clip_ucb_raw` 的 `ucb_lb`，預設 0.5（`config/eval/policy/default.yaml`，亦見 `config/vae.py:571`）。
  - `decision_rules.py:16` 寫死 `ucb_beta = 1.0`。
  - 目前只有 task success 主指標；damage / over-cut / 安全 margin 未獨立報告，且僅有平均，無 worst-case / tail。
- **P**（**E2-e: threshold + β sweep, worst-case / CVaR safety**）：
  1. `ucb_lb` sweep：{0.1, 0.3, 0.5, 0.7, 0.9, 0.99}。
  2. `ucb_beta` sweep：{0, 0.5, 1, 2, 3}（完整 grid）。
  3. 每個 setting 報：
     - **Safety (worst-case + tail)**：
       - **min** cut-plane-to-target-surface distance（worst-case，不是 mean）。
       - **95th-percentile damage volume (CVaR)**。
       - damage rate（mean，作為對照）。
     - **Efficiency**：移除體積、所需 cut 數。
     - **Conservativeness**：under-cut 率。
     - **Failure-mode taxonomy bar chart**（手檢失敗 episode 分類：wrong-axis / overshoot / premature-stop / target-misidentification）。
  4. 畫 damage-vs-removal Pareto，標示推薦操作點（與 W3 selection criterion 對齊）。
  5. 在 `decision_rules.py` 內加註：threshold 物理意義、β 選取理據（連結 W3）。
- **D**：
  - 圖：safety–efficiency Pareto + failure-mode bar chart。
  - 表：threshold × β 推薦值與依據。
  - 文字：Method 內補「Decision threshold rationale」段；W3 引 selection criterion；Limitation (W2) 註明 β 為 sweep-calibrated。

---

## 3. 文字修改（W1–W5）

### W1 — Abstract / Introduction / Conclusion 改寫（contribution-bullet rewrite + verb surgery）

**Verb constraint（嚴格 enforce）**：
- 允許：*formulates, introduces, analyzes, characterizes, shows in simulation*.
- **禁止**（明確 forbidden patterns，response 草稿與所有正文 scan 必過）：
  - "enable deployment" / "enables ... in industrial settings"
  - "toward [real-world / industrial / deployment / extraction in the field]"
  - "next stage" / "stepping stone" / "preliminary"
  - "future work alongside real-robot integration"
  - "calibration data future deployment needs"
  - "static teaser photo"（提案中曾出現，整批撤回）

**六條 verbatim contribution bullets**：

1. We **formulate** non-destructive internal-part extraction as an iterative loop coupling a conditional-diffusion estimator of internal voxel structure with a cutting-action planner over axis-aligned cuts.
2. We **introduce** a classifier-free-guidance conditional diffusion model that produces sample ensembles for per-voxel presence under partial observation, and **characterize** its sensitivity to the guidance scale ω across early/mid/late observation regimes.
3. We **formulate** a UCB-thresholded decision rule (parameters β, `ucb_lb`) for cut selection and **characterize** its safety–efficiency Pareto, including worst-case and CVaR damage statistics and a failure-mode taxonomy.
4. We **introduce** the `feature_provider` interface that separates perception from the planner contribution, and **show in simulation** the planner's behavior under three instantiations (`ColorRangeProvider`, `MaskCorruptionProvider`, `KMeansRenderProvider`) including a held-out-render perception swap.
5. We **analyze** the empirical convergence of the iterative estimation–planning loop on the uncut sub-volume and **show in simulation** that a calibrated stopping rule matches or improves over a fixed planning horizon on held-out episodes.
6. We **characterize** the DDIM-sampling speed–quality Pareto across voxel resolution and backbone (UNet / DiT / U-ViT), establishing the operating envelope under which the planner runs.

**Abstract surgery**：替換動詞 ("enable" → "we present and analyze"，"achieve" → "we characterize")；Conclusion 撤回任何 "ready for industrial dismantling" 類措辭。**限定子句位置（已 commit）**：
- "in simulation" — 出現在 **Abstract 首句**（首次描述方法時即限定）。
- "under axis-aligned cuts" — 出現在 **Introduction contributions 段落**（第一條 contribution bullet 的脈絡句中明寫）。
- "with color-attribute observation" — 出現在 **Conclusion 首句**（總結方法時與 simulation 限定子句並列）。

對應 AE-5。

### W2 — Limitation 章節（scope statements，不是 future-work promises）

合併以下幾條 limitation（**bullet 1 與 bullet 3 改寫為 scope statement**）：

1. **Evaluation venue**（bullet 1，scope，不是 future-work）：
   > "Evaluation is conducted in simulation, which we argue (Discussion §W4) is the appropriate venue for the contributions claimed."
2. 對觀測雜訊 / 切割誤差的健壯性僅在 E1 mask-domain corruption + 平移 plane perturbation 下驗證；**non-axis-aligned cuts are outside the cutting-action class modeled in this paper**（與 R3-1 同步）。
3. **Perception module 為 scope statement**（bullet 3，改寫）：
   > "Our planner is evaluated with `ColorRangeProvider`; performance under other perception modules is characterized by `KMeansRenderProvider` and `MaskCorruptionProvider` results. Selecting a perception module for a given hardware target is composition work outside the planner contribution."
4. iterative loop 收斂為 empirical observation，無 theoretical guarantee。
5. OOD 評估限於 **geometry-OOD**（color semantics 沿用 training）；color-permutation row 額外刻劃 color-identity 依賴。
6. DDIM 在高解析度的 sampling 成本由 E4-b operating envelope 表報出，envelope 之外的應用須由 integrator 自行驗證。
7. UCB β、threshold 為 sweep-calibrated empirical 值。

**新增 dial-back sentence（R5 amendment a，verbatim）**：

> "Real-robot validation of an integrated perception–planning–cutting stack is a natural extension that other groups, or our group in subsequent work, could pursue by composing modules around the `feature_provider` interface and the operating envelope characterized in §E1, §E2-e, §E4-b. Such extension is not required for the algorithmic and analytical contributions claimed in this paper."

對應 R3-1, R3-3, R3-7, R3-8, R3-9, R3-10, AE-5。

### W3 — Method 章節參數理據補充 + 預宣告 selection criterion

針對 voxel resolution、ω、`ucb_lb`、`ucb_beta`、`sampling_timesteps`、`task_step` 共 **6 項**，**統合於單一 consolidated subsection**（「Parameter selection rationale」），整節 **總頁數上限 ~1.5 頁**（避免六段各 ¼–½ 頁堆出 3+ 頁的 Method bloat — Applied Intelligence 有 page budget）。組織方式：以 selection-criterion table（R3-2 / E2-a）為節首，下方一個短段（≤3 句）一個參數，明引對應 criterion 的結果。每個參數的「選定值」需以該 criterion 的結果解釋，**不**得單純呈現「我們試了幾個值，這個 best」的表述。對應 AE-1、R3-2、R3-4、R3-5、R3-8、R3-10。

### W4 — 新 Discussion 子節：「Scope of Evaluation: Why Simulation is the Appropriate Venue」

**Four-anchor scaffold**（順序：load-bearing first）：

**Argument 1 — Contribution is an algorithm/formulation, not a system.**
- 主張：simulation 是 algorithm-class contributions 的標準 evaluation venue（如 MPC、TAMP、motion planning、diffusion-based planning）。
- **Citation anchors**：
  - Janner et al., *"Planning with Diffusion for Flexible Behavior Synthesis,"* ICML 2022（diffusion-planning 算法 contribution，sim-only 驗證）。**或** Chi et al., *"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,"* RSS 2023（替代 anchor，同 sim-first pattern）。
  - Garrett et al., *"Integrated Task and Motion Planning"* (PDDLStream), Annual Review of Control, Robotics, and Autonomous Systems, 2021（TAMP 算法 contribution，sim 驗證為標準）。
  - **Backup**：Williams, Aldrich, Theodorou, *"Model Predictive Path Integral Control,"* AIAA JGCD 2017（MPC algorithm contribution；保留為 backup mention，主 anchor 已換為 diffusion-planning works）。
  - **Backup #2**：Toussaint, *"Logic-Geometric Programming,"* IJCAI 2015 / IJRR 2018（同 sim-evaluation pattern）。

**Argument 2 — Separation of concerns via `feature_provider`.**
- 主張：perception / planning / execution 是獨立子領域；本論文 contribution 為 estimation+planning loop，perception 經 `feature_provider` 顯式介面外接。Physical execution **by design** 在 contribution scope 之外，**不是** deferral。
- 內部引用 R3-9 介面 + 三 instantiations。

**Argument 3 — Prior-art parity in dismantling / disassembly-sequence planning.**
- 主張：dismantling / disassembly-sequence-planning 領域標準 evaluation venue 為 simulation。
- **Citation anchor**：
  - 具體 paper：**`[PLACEHOLDER — author to resolve before submission]`** 一篇 Tao 或 Laili 作者線的 DSP paper（候選 venue：*Robotics and Computer-Integrated Manufacturing*, *CIRP Annals*, *Journal of Manufacturing Systems*；目標年份 2018–2024）。author 在撰稿前須 verify 確切 title / year / authors 並回填本檔；未 resolve 前 response letter 與正文皆不得 paste 該 citation。
  - Backup monograph：Vongbunyong & Chen, *Disassembly Automation*, Springer, 2015（field convention reference）。
  - **⚠ Strategy.md note**：「field standard is simulation」此一論斷必須在撰稿前以 recent disassembly survey（*Journal of Manufacturing Systems* 或 *CIRP Annals* 2020–2024）verify，**未 verify 前不得寫入正文**。若 survey 不支持該論斷則 Argument 3 退守為「multiple DSP works evaluate in simulation」（弱版）而非「field standard」。

**Argument 4 — Reproducibility.**
- 主張：part-extraction 規模的實機切割於跨 lab 不可重現（fixturing / blade / material / instrumentation 皆異）；controlled simulation 為 ω, β, `ucb_lb`, voxel resolution 提供固定 calibration substrate。
- 內部引用：E1 noise budget、E2-e safety Pareto、W5 specification surface 為 community 可直接 compose 的 calibration data。

### W5 — Specification-surface table（reframed deployment-pipeline table）

把舊的「deployment-pipeline sketch」**reframe 為 specification surface**：不是作者 roadmap，而是 integrator 讀表決定如何 compose。

| Component | 規格內容 |
|---|---|
| **Sensor class** | RGB-D（如 Intel RealSense D435i-class）作為 surface 來源；monocular RGB 作為 texture 來源。對應於本論文以 color attribute 作 surrogate observation 的位置。 |
| **Segmentation backbone slot** | 經 `feature_provider` 介面注入。本論文 instantiations 為 `ColorRangeProvider` / `MaskCorruptionProvider` / `KMeansRenderProvider`；學習式 segmentation（Mask2Former / SAM2）作為同介面的相容延伸。 |
| **Cutting hardware class** | Position-controlled saw / water-jet，planar end-effector trajectory。非軸對齊切割屬本論文未建模之 action class（見 R3-1 / W2）。 |
| **Expected dominant failure modes**（出處 = 本論文表格） | (i) mask false-positive blobs → over-cut（E1 表）；(ii) plane-translation Δ > 1 voxel → boundary damage（E1 表）；(iii) `ucb_lb` mis-calibration → under-cut stall（E2-e Pareto）。 |

**Framing 句（verbatim）**：

> "The experiments in this paper characterize the operating envelope an integrator must respect when composing perception, planning, and execution modules around the `feature_provider` interface."

對應 AE-4。

---

## 4. 工作排程建議（粗估）

| 階段 | 內容 | 預估時間 | 主要產出 |
|---|---|---|---|
| Week 1 | E1 mask-domain corruption 程式碼 + E2 sweep 腳手架（`scripts/eval/sweeps/`、`config/env/noisy.yaml`）+ `perception/feature_provider.py` 介面骨架 | 1 週 | sweep infra + 介面 |
| Week 2 | E2 全 sweep 跑完（ω, ucb_lb, ucb_beta, voxel res, sampling_steps, task_step） | 1 週（GPU bound） | sweep tables + selection-criterion 結果 |
| Week 3 | E3 audit + matched-compute VAEAC rerun + E4-b sampling profile | 1 週 | baseline table + Pareto |
| Week 4 | E1 noise ablation 全跑 + E4-a 外部 mesh OOD（含 color-permutation row）+ `KMeansRenderProvider` 跑（held-out fit set，neutral + harsh lighting） | 1 週 | robustness + OOD + perception swap |
| Week 5 | 圖表整理、W1–W5 文字撰寫、Response to Reviewers 草稿、forbidden-pattern scan | 1 週 | revision draft |

> 假設 1–2 張 RTX 3090。若僅 1 張卡，Week 2–4 順序跑，延長至 7–8 週。
>
> **Drop-first triage**：若 Week 4 slip，最先 drop 的是 KMeansRenderProvider 的 `texture_overlay` lighting condition（保留 `lighting_neutral` + `lighting_harsh`，harsh 已含 specular failure mode）。其次可 drop 的是 simple-pipeline 端的部分 generator 擴充項。E2 selection-criterion table 與 W4 / W5 撰寫不可 drop。

---

## 5. Repo 變更總覽（建議新增 / 修改）

```
新增：
  config/env/noisy.yaml                              # E1 mask-domain noise switch
  config/eval/sweeps/*.yaml                          # E2 sweep configs（單一矩陣）
  scripts/eval/sweeps/run_sweep_cfg.sh
  scripts/eval/sweeps/run_sweep_ucb.sh               # 同時 sweep ucb_lb 與 ucb_beta
  scripts/eval/sweeps/run_sweep_resolution.sh
  scripts/eval/sweeps/run_sweep_sampling_steps.sh
  scripts/eval/sweeps/run_sweep_horizon.sh
  scripts/eval/baseline_audit.py                     # E3 audit-all
  scripts/eval/profile_sampling.py                   # E4-b latency / memory / energy
  scripts/data_generation/ood_external_mesh.py       # E4-a 外部 mesh + 色彩 assignment
  scripts/data_generation/render_for_perception_test.py  # R3-9 PyVista render 兩 set
  denoising_diffusion_pytorch/perception/            # R3-9 — 主文 design contribution
    feature_provider.py                              # abstract base
    color_range_provider.py                          # 既有 color-mask 流程
    mask_corruption_provider.py                      # E1 corruption 入口
    kmeans_render_provider.py                        # K-means on rendered slice

修改：
  denoising_diffusion_pytorch/env/voxel_cut_sim_v1.py        # 平移 plane perturbation only
  denoising_diffusion_pytorch/eval/episode_runner.py         # uncut-subvolume 收斂指標 logging
  denoising_diffusion_pytorch/eval/episode_artifact_manager.py
  denoising_diffusion_pytorch/policy/decision/decision_rules.py  # β, threshold 物理意義註解 + W3 cross-ref
  config/inferencer/conditional_diffusion.yaml      # 暴露 sampling_steps 等供 sweep override
  config/eval/policy/default.yaml                   # 暴露 ucb_lb, ucb_beta, guidance_scale
```

`denoising_diffusion_pytorch/perception/` 屬主文 design contribution，**不**是 supplementary-only。

---

## 6. 對 reviewer / AE 回信的引用對照

| 回信段落 | 對應 Plan 條目 | 主要證據（圖 / 表 / 文字） |
|---|---|---|
| Response to AE-1 | E2-a + W3 selection-criterion table | 統一 sweep 表 + Method §參數理據 |
| Response to AE-2 | E2-b（uncut-subvolume Δ + 30/70 dev/test stopping rule） | 收斂曲線圖 + dev-calibrated stopping rule 在 test split 的 success / 步數對照表 |
| Response to AE-3 | E3 audit + matched-compute VAEAC rerun | audit 全 baseline 規格表 + VAEAC matched-compute 對照表 |
| Response to AE-4 | §7 verbatim 段落 + W4 + W5 + E1 + E2-e + E4-b + R3-9 三 provider | operating-envelope reframe，明引 W4 / W5 |
| Response to AE-5 | W1（六 contribution bullets + verb surgery）+ W2（dial-back sentence） | 改寫後的 Abstract / Introduction / Conclusion / Limitation |
| Response to R3-1 | E1（mask-domain corruption + Δ 平移）+ R3-1 plan 中 "non-axis-aligned cuts outside cutting-action class modeled" 明文 + W2 同句 | robustness 表 |
| Response to R3-2 | E2-a + W3（selection-criterion table） | sweep 總表 + Method 段 |
| Response to R3-3 | E2-b（uncut-subvolume Δ）+ dev/test stopping rule | 收斂指標、stopping rule test-split 結果、empirical-only framing |
| Response to R3-4 | E2-c（boundary IoU + surface F-score @ τ；real cap 344；simple floor 16） | resolution × boundary IoU / F-score |
| Response to R3-5 | E2-d（Brier-score 取代 reliability diagram；按 observation regime 切片） | ω sweep 圖 |
| Response to R3-6 | E3 audit-all + matched-compute VAEAC only | baseline 對齊表 |
| Response to R3-7 | E4-a 外部 mesh corpus + geometry-OOD caption + color-permutation row | OOD 三 row 表 |
| Response to R3-8 | E4-b（resolution × sampling_steps × backbone；energy = latency × measured power） | Pareto + DiT vs UNet 對比（須由 profile 重現 4.5× 宣稱） |
| Response to R3-9 | `feature_provider` 介面 + 三 instantiations（ColorRange / MaskCorruption / KMeansRender），全部主文 | 介面圖 + perception column 的結果表 |
| Response to R3-10 | E2-e（worst-case min distance + 95th-percentile CVaR damage + failure-mode taxonomy bar chart + 完整 β sweep） | safety–efficiency Pareto + failure-mode bar chart |
| Cross-cut: Scope-of-Evaluation 子節 | W4 four-anchor scaffold（Janner 2022 / Garrett 2021 / Tao DSP `[PLACEHOLDER — see §3 W4 Argument 3]` / Vongbunyong-Chen 2015） | 新 Discussion 子節 |
| Cross-cut: Specification surface | W5 表 | integrator-facing spec table |

---

## 7. Verbatim Response-to-AE-4 paragraph

> **Response to AE-4.** The AE writes: *"the authors should... provide stronger evidence regarding scalability and deployment feasibility under more realistic sensing and cutting conditions."* We address this by reinterpreting *deployment feasibility* as **characterizing the operating envelope** an integrator would require to compose our planner with a specific perception and cutting stack. Concretely: (i) §5.4 reports the noise budget (mask-domain flip rate, dilation/erosion magnitude, plane translational perturbation Δ) the planner tolerates without success-rate collapse; (ii) §5.5 reports the safety–efficiency Pareto, worst-case and 95th-percentile damage, and failure-mode taxonomy, giving the safety margins the integrator must respect; (iii) §5.8 reports DDIM sampling latency, peak GPU memory, and energy across resolution and backbone, giving the compute envelope; (iv) the new `feature_provider` interface (Method §4.5) factors perception out of the planner contribution explicitly, and we evaluate three providers — `ColorRangeProvider`, `MaskCorruptionProvider`, and a held-out-render `KMeansRenderProvider` — to characterize planner behavior under **perception modules the planner was not trained against**; (v) §5.9 characterizes the **geometric envelope** via external-mesh OOD with deterministic part-color assignment, with explicit color-semantics-inheritance caveat. The new Discussion subsection §6.2 *Scope of Evaluation: Why Simulation is the Appropriate Venue* argues, with anchors in diffusion-based planning and TAMP algorithm-class evaluation precedents and in disassembly-sequence-planning prior-art parity, that simulation is the appropriate venue for the contributions claimed; the moderated contributions in the rewritten Abstract and Introduction make no claim about a physical cutting system.

---

## 8. Post-coherence-audit amendments

After three independent paper-coherence reviews (A: narrative flow, B: contribution-claim integrity, C: reviewer-satisfaction simulation), the following amendments are required **in addition to** §0.5–§7. Where conflicts arise, §8 takes precedence.

### 8.1 ⚠ CRITICAL — delete original §6.2 "Real-World Deployment"

The submitted manuscript's §6.2 (PDF p.18–20) is built on forward-deployment language ("Applying the proposed method to real environments is an interesting direction for future work", "may enable the proposed loop to operate under realistic visual variability", etc.). It is incompatible with the new W4 "Scope of Evaluation" subsection. **If left in place, the paper contradicts itself within four pages.**

**Action**: delete the original §6.2 entirely. Reallocate its content:

- Abrasive-waterjet / diamond-wire discussion → §6.3 W5 Specification-surface table, "Cutting hardware class" row.
- SAM / Mask-R-CNN / learned-segmentation paragraph → §4.6 (feature_provider) as one "instantiation-pathway sentence" describing how learned segmenters compose into the interface.
- "Future work" sentences → struck entirely.

### 8.2 ⚠ CRITICAL — feature_provider needs framework hook in §4.1

§4.6 (feature_provider) currently lands after §4.4 with no antecedent in the framework description. Reader meets the abstraction without prior conceptual hook.

**Action**:

- Add one sentence at the end of §4.1 framework paragraph: "Stage (3) deployment consumes observations through a `feature_provider` interface (§4.6), making perception explicit and swappable."
- Update Figure 2 (framework diagram) to show the `feature_provider` slot between observation and planner.
- Without this hook, §4.6 reads as a retrofit and W1 Bullet 4 ("introduce the feature_provider interface") looks like a response artifact rather than a design choice.

### 8.3 ⚠ CRITICAL — resolve Anchor 3 citation gate

Strategy §3 W4 Argument 3 has the Tao/Laili DSP citation marked `[PLACEHOLDER — author to resolve before submission]`. Reviewer C audit predicts this is a single most-damaging untreated item if left unresolved.

**Action before §6.2 prose is finalized**:

1. Read one specific candidate paper from Tao or Laili (in RCIM or CIRP Annals, 2020–2024) in full.
2. Verify the "field standard is simulation" claim against the actual survey text — do not paraphrase from memory.
3. If verified → cite specifically with year and venue.
4. If **not** verified → reword Argument 3 from "prior-art parity" to the weaker form "representative works also adopt simulation evaluation," and accept that an AE familiar with disassembly literature (where Vongbunyong-Chen / CIRP have a hardware-validation tradition) may not concede the parity claim.

### 8.4 §4.6 framing — software-factoring, **not** perception study

W1 Bullet 4 "introduce feature_provider interface" risks being read as a perception contribution. Audit C predicts reviewer pushback: "the three providers all consume rendered or synthetic inputs; this is a software factoring, not a perception study."

**Action**:

- §4.6 opening paragraph must state explicitly: *"This section contributes a software abstraction that factors perception out of the planner; it is not a perception study. The contribution is the interface and the demonstration that three substantively different providers slot into it without changing the planner."*
- Cross-reference §6.1 Limitation scope statement ("Selecting a perception module for a given hardware target is composition work outside the planner contribution") to make the boundary explicit.
- §3.6 in the original strategy is hereby renamed to §4.6 (see §8.5 below).

### 8.5 Final section-number assignment

Original paper structure is §3 Preliminary, §4 Proposed Method (with §4.1 Framework, §4.2 Voxel Representation, §4.3 Internal Structure Estimation, §4.4 Cutting Action Planning), §5 Experiments, §6 Discussion (with §6.1 Initial Cutting Selection, §6.2 Real-World Deployment — **deleted per §8.1**), §7 Conclusion. Earlier strategy used floating "§3.5 / §3.6" / "§4.5 / §4.6" / "§4.7 / §4.8" which conflict with the existing §4 Method subsections. Final numbering (clean, conflict-free):

| Strategy ref | Final section |
|---|---|
| Existing Method | §4.1 Framework / §4.2 Voxel Rep / §4.3 Internal Structure Estimation / §4.4 Cutting Action Planning **(unchanged)** |
| Feature-provider abstraction (was strategy §3.6) | **§4.5** (NEW, after existing §4.4) |
| `feature_provider` hook in framework | §4.1 paragraph + Fig. 2 update (per §8.2) |
| Parameter-rationale subsection (W3) | **§5.1.6** (table inside Common Experimental Setup; see §8.6) |
| CFG sweep (E2-d) | **§5.2** |
| Convergence + stopping rule (E2-b) | **§5.3** |
| Robustness ablation (E1) | **§5.4** |
| Safety analysis (E2-e) | **§5.5** |
| Voxel resolution sweep (E2-c) | **§5.6** |
| Matched-compute VAEAC (E3) | **§5.7** |
| DDIM scalability (E4-b) | **§5.8** |
| OOD evaluation (E4-a) | **§5.9** |
| Scope of Evaluation (W4) | **§6.2** (replaces deleted original §6.2) |
| Specification-surface table (W5) | **§6.3** |
| Limitation (W2) | **§6.4** (consolidated, 4 bullets per §8.7) |

**NB**: this is the authoritative numbering. Any conflict with §2–§7 of this strategy doc is resolved by §8.5. **All references to §3.5 / §3.6 / §4.6+ / §4.7 / §4.8 in earlier sections and §7 must be reread under this mapping.**

### 8.6 Move parameter-rationale (W3) into §5.1.6 as table

§4.5 (W3) placed in Method causes forward-references to dev/test splits, Brier scores, and Pareto knees that are introduced only in Experiments. Audit A recommends relocation.

**Action**: place W3 as **§5.1.6 "Parameter selection rationale"** at end of Common Experimental Setup. Format: the Table 4 pre-declared selection-criterion table plus ≤1 page of caption-style prose. This (a) keeps selection criteria next to the sweeps that justify them, (b) saves ~1 page of Method bloat, (c) preserves the W3 commitment because the table is the load-bearing object.

### 8.7 Trim Limitation (§6.4) from 7 bullets to 4

Audit A flags redundancy between Limitation and §6.2 / §4.6 — the same claim would appear in three places. Trim to:

1. Non-axis-aligned cuts outside the cutting-action class modeled in this paper.
2. Noise tolerance characterized in §4.3 — bounds, not certifications.
3. Geometry-OOD scope: color semantics inherited from training-time conventions (§5.7).
4. Sweep-calibrated thresholds are reproducibility artifacts, not safety certificates.

The R5 dial-back sentence on real-robot extension is **moved** to the closing of §6.2 (positive defense), not Limitation.

### 8.8 Subsume original Tables 1 & 3 into new ucb_lb sweep

Original η = {0.0, 0.5, 1.0} ablation (Tables 1 & 3 in the submitted PDF) is a degenerate sub-grid of the new `ucb_lb ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 0.99}` sweep (R3-10 / E2-e).

**Action**: delete original Tables 1 & 3; the new safety-analysis table (§4.5 Table 9) carries the same information at finer granularity.

### 8.9 Operating-envelope: 5-artifact framing (not 4)

Audit B flags that **geometry envelope** is a 5th axis the integrator needs but it is not in the AE-4 four-artifact list. §4.8 (now §5.7 per §8.5) already characterizes this axis.

**Action**: §7 Response-to-AE-4 paragraph is updated above to list five artifacts (i)–(v) including the new (v) §4.8 / §5.7 geometric envelope. W1 Bullet 6 "operating envelope" phrasing is now honest (it covers noise, safety, compute, perception, and geometry axes).

### 8.10 Pre-register contingency wordings for W1 Bullets 3, 5, 6

Audit B flags that Bullets 3, 5, 6 contain claims that are honest-result-conditional. Pre-register fallback wordings before sweeps run, so that no last-minute scramble happens if results don't support the stronger phrasing:

| Bullet | Strong wording (default) | Fallback wording (use if results don't support) |
|---|---|---|
| 3 | "characterize its safety–efficiency **Pareto**" | "characterize **how the safety–efficiency trade-off and dominant failure modes depend on β and `ucb_lb`**" (use if β × ucb_lb is monotone dominated, not Pareto-shaped — keeps failure-mode taxonomy as the load-bearing contribution regardless of Pareto shape) |
| 5 | "**matches or improves over** a fixed planning horizon on held-out episodes" | "**characterizes the trade-off between** calibrated stopping and fixed-horizon planning" (use if dev-calibrated rule loses on test split) |
| 6 | "establishing the operating envelope **under which** the planner runs" | "characterizing the operating envelope **within which the planner is shown to run**" (use if 344 fails latency floor across all backbones) |

The Abstract and Introduction contribution bullets must be drafted with the strong wording but reviewed against actual results before final submission.

### 8.11 Pre-submission verb-leak watchlist (5 surfaces)

Audit B identifies five surfaces where the verb constraint can leak in non-contribution-bullet prose:

| Section | Risk pattern (do NOT use) | Required wording |
|---|---|---|
| §6.2 envelope statement | "practical limits" | "latency budget characterized in §4.7" |
| §4.3 / §5 robustness narrative | "achieves robustness up to" | "success rate remains above X up to" |
| §5.5 matched-compute VAEAC table | "outperforms VAEAC under matched compute" | numerical statement only; no comparative verb |
| §5 Table 5 caption (stopping rule) | "achieves higher success than fixed horizon" | "matches or improves" (or §8.10 fallback) |
| §4.6 feature_provider | "enables deployment with learned segmentation" | "compose into the same interface as a natural extension" |

Author must scan for these surfaces in the final manuscript draft before submission.

### 8.12 R3-1 / R3-6 / R3-9 specific reviewer-pushback mitigations

Audit C predicts pushback on these three items. Mitigations:

- **R3-1**: Add one sentence in §4.3 citing literature-justified noise parameters (e.g., RealSense / RGB-D segmentation noise rate from a 2022+ benchmark) to ground the mask-corruption rates. Without this grounding the corruption model is self-graded.
- **R3-6**: §5.5 matched-compute VAEAC table caption must include the scope rationale verbatim: *"The audit (§5.1 Table 2) shows PCD-DM and CVAE are within 1.2× compute of the proposed method; matched-compute is applied to VAEAC as the baseline named explicitly by Reviewer #3. The within-1.2× margin is documented in Table 2."* This defense is load-bearing for R3-6 pushback.
- **R3-9**: §4.6 must explicitly state software-factoring scope per §8.4 above.

### 8.14 ⚠ Config canonicalization — `config/vae.py` and `config/vae_simple_model.py` override YAMLs

Final-audit Dimension 1 finding: load-bearing defaults for `ucb_lb=0.5`, `cfg_omega=0.2`, `task_step=8` live in **Python config files** (`config/vae.py:565–699` and `config/vae_simple_model.py:492–574`), not only in the YAMLs `config/eval/policy/default.yaml` and `config/inferencer/conditional_diffusion.yaml`. A sweep that only overrides YAML will silently miss these Python paths.

**Action** (choose one before E2 sweep harness starts):

- **Option A (preferred)**: Migrate the defaults from `config/vae.py` and `config/vae_simple_model.py` into the YAMLs, then delete the Python overrides. Cleaner long-term.
- **Option B**: Update strategy.md §5 "Repo 變更總覽" "modify" list to explicitly include `config/vae.py` and `config/vae_simple_model.py` alongside the YAMLs, and patch both call paths in every sweep run.

Additionally: `denoising_diffusion_pytorch/policy/decision/decision_rules.py:15` hardcodes `ucb_beta = 1.0`. The R3-10 β sweep requires turning this into a Hydra-overridable parameter (~1 hour work). Add this as a sub-task of E2-e in §5 "modify" list.

### 8.15 ⚠ `voxel_cut_handler` Δ-extension is non-trivial — explicit scoping

Final-audit Dimension 1 finding: `denoising_diffusion_pytorch/env/voxel_cut_sim_v1.py:115` `voxel_cut_handler.__init__` takes `(grid_config, mesh_components, zero_initialize, pre_near_by_cells)` — **no Δ offset parameter**. R3-1 step 2 (translational plane perturbation `Δ ∈ {0, ±1, ±2}`) understates the change.

**Action**: extend the cutter as an explicit sub-task in §2 R3-1 implementation:

1. Add `delta_offset: int = 0` to the action dict consumed by `voxel_cut_handler.get_obs` (line ~164) and `update_color` (line ~183).
2. After `index_map_fn.map_1d_to_2d_loc(action['loc'])` returns `(i, j)`, apply the offset along the slicing axis *before* extracting the box-slice: `(i + delta_offset, j)` for x-axis cuts; analog for y/z.
3. Clamp the offset to grid bounds; if out-of-bounds, log and proceed with Δ=0 (silent failure protection).
4. Default 0 preserves all existing eval-pipeline behavior bit-exact.

Estimated effort: ~½ day code + 1 hour testing on a fixed eval episode to verify Δ=0 reproduces baseline numbers. Without explicit scoping, an engineer will conflate axis-Δ with the existing discrete `loc` and produce a no-op perturbation.

### 8.13 Schedule impact

The §8 amendments are **structural and rhetorical**, not new experiments. Net effect on the 5–7 week schedule:

- §8.1, §8.2, §8.4, §8.6, §8.7, §8.8 → writing-phase edits, no GPU cost. Estimated +0.5–1 day of writing.
- §8.3 → 0.5 day verification reading.
- §8.5 → renumbering pass, ~2 hours.
- §8.10 → register fallbacks before sweep results are in, ~1 hour.
- §8.11, §8.12 → final-draft scan, ~half day.
- §8.9 → already reflected in updated §7 paragraph above; one extra line in Abstract or Introduction summarizing the 5-axis envelope.

**Total**: ~2–3 days of writing-phase work absorbed into Week 5.
