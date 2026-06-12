# Analysis scripts

The analysis scripts are organized by the type of map or metric they visualize.

## `presence_frequency_maps/`

Fig.10-style spatial visualizations derived from `raw_pred_image` ensembles.

These scripts show voxel-level target-part presence frequency maps. A value at a
voxel means the fraction of generated samples in which that voxel is classified
as the target part. These maps are useful for interpreting the spatial
distribution of generated target-part locations.

Recommended entry points:

```bash
python -m scripts.analysis.presence_frequency_maps.plot_raw_pred_presence_frequency_maps \
  --manifest path/to/manifest.csv \
  --out_path path/to/presence_frequency.pdf
```

```bash
python -m scripts.analysis.presence_frequency_maps.plot_cost_log_aligned_presence_frequency_maps \
  --cost_map_log path/to/0_cost_map_logs.pickle \
  --out_path path/to/cost_log_aligned_presence_frequency.pdf
```

```bash
python -m scripts.analysis.presence_frequency_maps.plot_raw_pred_presence_frequency_with_visual_3d_dilation \
  --rollout_root path/to/epsilon_greedy_00 \
  --case Object_A \
  --episode 0 \
  --out_path path/to/presence_frequency_with_3d_dilation.pdf
```

## `axis_score_maps/`

Policy-level axis-wise score visualizations derived from `cost_ensembles` in
`*_cost_map_logs.pickle`.

These scripts visualize the intermediate score maps used by `clip_ucb_raw`,
namely the ensemble mean plus one standard deviation after the 1D safety-margin
maximum filter, or the final thresholded decision map. They are not voxel-level
Fig.10-style presence-frequency maps.

Recommended entry point:

```bash
python -m scripts.analysis.axis_score_maps.plot_policy_axis_maps \
  --cost_map_log path/to/0_cost_map_logs.pickle \
  --target blue \
  --ucb_lb 0.5 \
  --radii 0,1,2 \
  --score_mode ucb \
  --out_path path/to/policy_axis_scores.pdf
```

## Legacy flat scripts

The original flat scripts remain available for backward compatibility:

- `plot_presence_score_maps.py`
- `plot_cost_log_presence_score_maps.py`
- `plot_safety_margin_presence_maps.py`
- `plot_safety_margin_decision_maps.py`

New commands should prefer the grouped entry points above.
