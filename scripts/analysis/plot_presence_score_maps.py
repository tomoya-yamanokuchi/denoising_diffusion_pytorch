from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image


DEFAULT_COLOR_RANGES = {
    "blue": ((0, 0, 250), (5, 5, 255)),
    "red": ((250, 0, 0), (255, 5, 5)),
    "green": ((0, 250, 0), (5, 255, 5)),
    "yellow": ((250, 250, 0), (255, 255, 5)),
    "simple_blue": ((0, 140, 140), (115, 255, 255)),
    "simple_red": ((179, 26, 26), (255, 102, 102)),
    "simple_yellow": ((179, 179, 26), (255, 255, 204)),
}
DEFAULT_VIEW_SPECS = [("Side view", "x", -1), ("Top view", "z", 2)]


@dataclass(frozen=True)
class MethodSpec:
    label: str
    root: Path
    source: str
    case: str | None = None
    episode: int | None = None
    step: int | None = None


@dataclass(frozen=True)
class ViewSpec:
    label: str
    projection_axis: str
    rot90: int


@dataclass
class PanelData:
    score: np.ndarray
    mask: np.ndarray | None
    is_ground_truth: bool


def parse_optional_int(row: dict[str, str], key: str, *, row_no: int, manifest_path: Path) -> int | None:
    text = (row.get(key) or "").strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer for {key!r} at manifest row {row_no}: {text!r} in {manifest_path}"
        ) from exc


def parse_manifest(path: Path) -> list[MethodSpec]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    if "rollout_root" not in rows[0]:
        raise ValueError(f"Manifest must contain a rollout_root column. Available columns: {list(rows[0].keys())}")

    specs = []
    for idx, row in enumerate(rows):
        row_no = idx + 2
        root_text = (row.get("rollout_root") or "").strip()
        if not root_text:
            raise ValueError(f"Empty rollout_root at manifest row {row_no}: {path}")
        label = str(row.get("method") or row.get("label") or row.get("name") or Path(root_text).name).strip()
        if not label:
            raise ValueError(f"Empty method label at manifest row {row_no}: {path}")
        source = str(row.get("source") or "").strip().lower()
        if not source:
            lower = label.lower()
            source = "oracle" if ("ground" in lower or "gt" in lower or "oracle" in lower) else "raw_pred"
        if source not in {"raw_pred", "oracle"}:
            raise ValueError(f"Unsupported source={source!r} at manifest row {row_no}. Use raw_pred or oracle.")
        specs.append(
            MethodSpec(
                label=label,
                root=Path(root_text),
                source=source,
                case=(row.get("case") or "").strip() or None,
                episode=parse_optional_int(row, "episode", row_no=row_no, manifest_path=path),
                step=parse_optional_int(row, "step", row_no=row_no, manifest_path=path),
            )
        )
    return specs


def parse_case_filter(text: str | None) -> list[str] | None:
    return None if text is None or text.strip() == "" else [x.strip() for x in text.split(",") if x.strip()]


def parse_view_specs(text: str | None) -> list[ViewSpec]:
    if text is None or text.strip() == "":
        return [ViewSpec(*x) for x in DEFAULT_VIEW_SPECS]
    specs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid view spec {item!r}. Use 'label:projection_axis:rot90'.")
        label, axis, rot = parts
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported projection axis: {axis!r}")
        specs.append(ViewSpec(label, axis, int(rot)))
    if not specs:
        raise ValueError("No valid view specs were parsed.")
    return specs


def ordered_method_labels(method_specs: list[MethodSpec]) -> list[str]:
    labels, seen = [], set()
    for spec in method_specs:
        if spec.label not in seen:
            labels.append(spec.label)
            seen.add(spec.label)
    return labels


def resolve_method_spec_for_case(method_specs: list[MethodSpec], *, label: str, case: str) -> MethodSpec:
    exact = [s for s in method_specs if s.label == label and s.case == case]
    shared = [s for s in method_specs if s.label == label and s.case is None]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"Multiple manifest rows found for method={label!r}, case={case!r}")
    if len(shared) == 1:
        return shared[0]
    if len(shared) > 1:
        raise ValueError(f"Multiple shared manifest rows found for method={label!r}. Use a case column.")
    raise ValueError(f"No manifest row found for method={label!r}, case={case!r}.")


def build_presence_colormap(name: str):
    if name == "jet_bright":
        base = plt.get_cmap("jet")
        colors = base(np.linspace(0.0, 0.89, 256))
        colors[-1, :3] = np.asarray([1.0, 0.0, 0.0])
        return LinearSegmentedColormap.from_list("jet_bright", colors)
    return plt.get_cmap(name)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def infer_side_length_from_image(image: np.ndarray) -> int:
    height, width = image.shape[:2]
    if height != width:
        raise ValueError(f"Only square tiled voxel images are supported. Got height={height}, width={width}.")
    for side in range(2, height + 1):
        tile_grid = int(round(math.sqrt(side)))
        if tile_grid * tile_grid == side and side * tile_grid == height:
            return side
    raise ValueError(f"Could not infer voxel side length from image side={height}. Pass --side_length.")


def tiled_image_to_cubic(image: np.ndarray, side_length: int | None = None) -> np.ndarray:
    k = int(side_length or infer_side_length_from_image(image))
    if image.shape[0] % k != 0 or image.shape[1] % k != 0:
        raise ValueError(f"Image shape {image.shape[:2]} is not divisible by side_length={k}.")
    tile_grid_x = image.shape[1] // k
    if (image.shape[0] // k) * tile_grid_x < k:
        raise ValueError(f"The tiled image does not contain enough KxK tiles for side_length={k}.")
    cubic_shape = (k, k, k) if image.ndim == 2 else (k, k, k, image.shape[-1])
    cubic = np.zeros(cubic_shape, dtype=image.dtype)
    for z in range(k):
        tile_y, tile_x = divmod(z, tile_grid_x)
        tile = image[tile_y * k : (tile_y + 1) * k, tile_x * k : (tile_x + 1) * k]
        cubic[k - 1 :: -1, :, z] = tile
    return cubic


def color_mask(image: np.ndarray, color_name: str) -> np.ndarray:
    if color_name not in DEFAULT_COLOR_RANGES:
        raise ValueError(f"Unknown color_name={color_name!r}. Choose from {sorted(DEFAULT_COLOR_RANGES)}.")
    lb, ub = DEFAULT_COLOR_RANGES[color_name]
    return np.all((image >= np.asarray(lb, dtype=np.uint8)) & (image <= np.asarray(ub, dtype=np.uint8)), axis=-1)


def non_background_mask(cubic_rgb: np.ndarray) -> np.ndarray:
    arr = cubic_rgb.astype(np.int16)
    return ~(np.all(arr >= 245, axis=-1) | np.all(arr <= 5, axis=-1))


def project_volume(volume: np.ndarray, projection_axis: str, rot90: int) -> np.ndarray:
    projected = np.max(volume, axis={"x": 0, "y": 1, "z": 2}[projection_axis])
    return np.rot90(projected, rot90) if rot90 else projected


def resolve_step_dir(raw_pred_root: Path, step: int | None) -> Path:
    step_dirs = [p for p in raw_pred_root.iterdir() if p.is_dir() and p.name.startswith("step_")]
    if not step_dirs:
        raise FileNotFoundError(f"No step_* directories were found in: {raw_pred_root}")

    def step_index(path: Path) -> int:
        match = re.search(r"step_(-?\d+)$", path.name)
        return int(match.group(1)) if match else -10**9

    step_dirs = sorted(step_dirs, key=step_index)
    if step is None or step < 0:
        return step_dirs[-1]
    wanted = raw_pred_root / f"step_{step}"
    if not wanted.exists():
        raise FileNotFoundError(f"Requested step directory does not exist: {wanted}. Available: {[p.name for p in step_dirs]}")
    return wanted


def read_raw_pred_score_map(episode_dir: Path, *, step: int | None, target_color: str, side_length: int | None) -> np.ndarray:
    raw_pred_root = episode_dir / "raw_pred_image"
    if not raw_pred_root.exists():
        raise FileNotFoundError(f"raw_pred_image directory was not found: {raw_pred_root}")
    sample_paths = sorted(resolve_step_dir(raw_pred_root, step).glob("*.png"))
    if not sample_paths:
        raise FileNotFoundError(f"No PNG samples were found in: {raw_pred_root}")
    masks = []
    for sample_path in sample_paths:
        cubic = tiled_image_to_cubic(load_rgb(sample_path), side_length=side_length)
        masks.append(color_mask(cubic, target_color).astype(np.float32))
    return np.mean(np.stack(masks, axis=0), axis=0)


def find_oracle_image(episode_dir: Path) -> Path:
    for name in ["oracle_obs_cast_z_axis0.png", "oracle_obs_cast_z_axis_0.png", "oracle_obs_cast_z.png"]:
        candidate = episode_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(episode_dir.glob("oracle_obs_cast_z*.png"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No oracle_obs_cast_z*.png was found in: {episode_dir}")


def read_oracle_score_map(episode_dir: Path, *, target_color: str, side_length: int | None) -> np.ndarray:
    cubic = tiled_image_to_cubic(load_rgb(find_oracle_image(episode_dir)), side_length=side_length)
    return color_mask(cubic, target_color).astype(np.float32)


def read_shape_mask(episode_dir: Path, *, side_length: int | None) -> np.ndarray | None:
    try:
        cubic = tiled_image_to_cubic(load_rgb(find_oracle_image(episode_dir)), side_length=side_length)
    except FileNotFoundError:
        return None
    return non_background_mask(cubic).astype(bool)


def resolve_episode_dir(root: Path, case: str, episode: int) -> Path:
    if root.name.startswith("episode_"):
        return root
    case_root = root / case
    if not case_root.exists():
        if root.name == case:
            case_root = root
        else:
            raise FileNotFoundError(f"Case directory was not found: {case_root}")
    episode_dir = case_root / f"episode_{episode}"
    if episode_dir.exists():
        return episode_dir
    episodes = sorted(p for p in case_root.glob("episode_*") if p.is_dir())
    if not episodes:
        raise FileNotFoundError(f"No episode_* directories were found in: {case_root}")
    if episode < 0:
        return episodes[0]
    raise FileNotFoundError(f"Requested episode_{episode} was not found in {case_root}. Available: {[p.name for p in episodes]}")


def discover_cases(method_specs: list[MethodSpec], case_filter: list[str] | None) -> list[str]:
    if case_filter is not None:
        return case_filter
    explicit = [s.case for s in method_specs if s.case is not None]
    if explicit:
        return sorted(set(explicit))
    cases: set[str] = set()
    for spec in method_specs:
        if not spec.root.exists():
            continue
        for child in spec.root.iterdir():
            if child.is_dir() and not child.name.startswith("episode_") and any(p.is_dir() and p.name.startswith("episode_") for p in child.iterdir()):
                cases.add(child.name)
    if not cases:
        raise ValueError("Could not discover case names automatically. Please pass --case_filter.")
    return sorted(cases)


def crop_bounds_from_masks(masks: list[np.ndarray | None], padding: int) -> tuple[int, int, int, int] | None:
    valid = [m for m in masks if m is not None and np.any(m)]
    if not valid:
        return None
    union = np.zeros_like(valid[0], dtype=bool)
    for mask in valid:
        if mask.shape == union.shape:
            union |= mask
    ys, xs = np.where(union)
    if len(xs) == 0:
        return None
    return (max(int(ys.min()) - padding, 0), min(int(ys.max()) + padding + 1, union.shape[0]), max(int(xs.min()) - padding, 0), min(int(xs.max()) + padding + 1, union.shape[1]))


def apply_crop(image: np.ndarray, mask: np.ndarray | None, bounds: tuple[int, int, int, int] | None) -> tuple[np.ndarray, np.ndarray | None]:
    if bounds is None:
        return image, mask
    y0, y1, x0, x1 = bounds
    return image[y0:y1, x0:x1], None if mask is None else mask[y0:y1, x0:x1]


def pad_to_square(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = image.shape[:2]
    size = max(h, w)
    if h == size and w == size:
        return image, mask
    pt, pl = (size - h) // 2, (size - w) // 2
    pad = ((pt, size - h - pt), (pl, size - w - pl))
    out_image = np.pad(image, pad, mode="constant", constant_values=0.0)
    out_mask = None if mask is None else np.pad(mask, pad, mode="constant", constant_values=False)
    return out_image, out_mask


def render_score_panel(ax, score: np.ndarray, shape_mask: np.ndarray | None, *, ground_truth: bool, presence_cmap, background_mode: str, ground_truth_background_mode: str, ground_truth_style: str, panel_frame: bool) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    shape_mask = np.ones_like(score, dtype=bool) if shape_mask is None else shape_mask

    if ground_truth:
        target = score > 0.5
        if ground_truth_style == "target_only":
            rgb = np.ones((*score.shape, 3), dtype=float) * 0.88
            rgb[target] = np.asarray([0.25, 0.25, 0.25])
        else:
            if ground_truth_background_mode == "low_score":
                rgb = np.broadcast_to(np.asarray(presence_cmap(0.0)[:3], dtype=float), (*score.shape, 3)).copy()
            elif ground_truth_background_mode == "light_gray":
                rgb = np.ones((*score.shape, 3), dtype=float) * 0.92
            elif ground_truth_background_mode == "white":
                rgb = np.ones((*score.shape, 3), dtype=float)
            else:
                raise ValueError(f"Unknown ground_truth_background_mode={ground_truth_background_mode!r}")
            rgb[shape_mask] = np.asarray([0.86, 0.86, 0.86])
            rgb[target & shape_mask] = np.asarray([0.25, 0.25, 0.25])
    else:
        colored = presence_cmap(np.clip(score, 0.0, 1.0))[..., :3]
        if background_mode == "low_score":
            rgb = np.broadcast_to(np.asarray(presence_cmap(0.0)[:3], dtype=float), (*score.shape, 3)).copy()
        elif background_mode == "light_gray":
            rgb = np.ones((*score.shape, 3), dtype=float)
            rgb[~shape_mask] = np.asarray([0.92, 0.92, 0.92])
        elif background_mode == "white":
            rgb = np.ones((*score.shape, 3), dtype=float)
        else:
            raise ValueError(f"Unknown background_mode={background_mode!r}")
        rgb[shape_mask] = colored[shape_mask]

    ax.imshow(rgb, interpolation="nearest")
    for spine in ax.spines.values():
        spine.set_visible(panel_frame)
        if panel_frame:
            spine.set_linewidth(0.4)
            spine.set_edgecolor("0.5")


def add_outside_colorbar(fig, axes, presence_cmap, args) -> None:
    fig.tight_layout(rect=[0.0, 0.0, args.colorbar_layout_right, 1.0])
    positions = [ax.get_position() for ax in axes.ravel()]
    right_edge = max(pos.x1 for pos in positions)
    bottom_edge = min(pos.y0 for pos in positions)
    top_edge = max(pos.y1 for pos in positions)
    total_height = top_edge - bottom_edge
    cbar_height = total_height * args.colorbar_shrink
    cbar_bottom = bottom_edge + (total_height - cbar_height) / 2.0
    cbar_left = min(right_edge + args.colorbar_pad, 0.98 - args.colorbar_width)
    cax = fig.add_axes([cbar_left, cbar_bottom, args.colorbar_width, cbar_height])
    sm = plt.cm.ScalarMappable(cmap=presence_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Presence score", fontsize=args.label_fontsize)
    cbar.ax.tick_params(labelsize=args.label_fontsize - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--case_filter", type=str, default=None)
    parser.add_argument("--episode", type=int, default=0, help="Fallback episode index used when a manifest row has no episode column/value.")
    parser.add_argument("--step", type=int, default=-1, help="Fallback step index used when a manifest row has no step column/value. Negative means the last available step.")
    parser.add_argument("--target_color", type=str, default="blue", choices=sorted(DEFAULT_COLOR_RANGES))
    parser.add_argument("--side_length", type=int, default=None)
    parser.add_argument("--view_specs", type=str, default=None)
    parser.add_argument("--presence_cmap", type=str, default="jet_bright")
    parser.add_argument("--background_mode", type=str, default="white", choices=["white", "low_score", "light_gray"])
    parser.add_argument("--ground_truth_background_mode", type=str, default="white", choices=["white", "low_score", "light_gray"])
    parser.add_argument("--ground_truth_style", type=str, default="structure", choices=["structure", "target_only"])
    parser.add_argument("--crop_scope", type=str, default="case_view", choices=["case_view", "panel"])
    parser.add_argument("--no_square_panels", action="store_true")
    parser.add_argument("--panel_frame", action="store_true")
    parser.add_argument("--crop_padding", type=int, default=2)
    parser.add_argument("--no_auto_crop", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title_fontsize", type=int, default=11)
    parser.add_argument("--label_fontsize", type=int, default=10)
    parser.add_argument("--add_colorbar", action="store_true", help="Show a colorbar. Kept for backward compatibility.")
    parser.add_argument("--show_colorbar", action="store_true", help="Show a colorbar outside the panel grid.")
    parser.add_argument("--colorbar_layout_right", type=float, default=0.86)
    parser.add_argument("--colorbar_pad", type=float, default=0.025)
    parser.add_argument("--colorbar_width", type=float, default=0.018)
    parser.add_argument("--colorbar_shrink", type=float, default=0.45)
    args = parser.parse_args()

    method_specs = parse_manifest(args.manifest)
    method_labels = ordered_method_labels(method_specs)
    cases = discover_cases(method_specs, parse_case_filter(args.case_filter))
    view_specs = parse_view_specs(args.view_specs)
    presence_cmap = build_presence_colormap(args.presence_cmap)

    panel_data: dict[tuple[int, int, int], PanelData] = {}
    for case_idx, case in enumerate(cases):
        for col_idx, label in enumerate(method_labels):
            spec = resolve_method_spec_for_case(method_specs, label=label, case=case)
            episode = spec.episode if spec.episode is not None else args.episode
            step = spec.step if spec.step is not None else args.step
            episode_dir = resolve_episode_dir(spec.root, case, episode)
            shape_mask = read_shape_mask(episode_dir, side_length=args.side_length)
            if spec.source == "oracle":
                score_volume = read_oracle_score_map(episode_dir, target_color=args.target_color, side_length=args.side_length)
                is_gt = True
            else:
                score_volume = read_raw_pred_score_map(episode_dir, step=None if step < 0 else step, target_color=args.target_color, side_length=args.side_length)
                is_gt = False
            for view_idx, view in enumerate(view_specs):
                score = project_volume(score_volume, view.projection_axis, view.rot90)
                mask = None if shape_mask is None else project_volume(shape_mask.astype(float), view.projection_axis, view.rot90) > 0
                panel_data[(case_idx, col_idx, view_idx)] = PanelData(score, mask, is_gt)

    if not args.no_auto_crop:
        for case_idx, _case in enumerate(cases):
            for view_idx, _view in enumerate(view_specs):
                if args.crop_scope == "case_view":
                    bounds = crop_bounds_from_masks([panel_data[(case_idx, col_idx, view_idx)].mask for col_idx in range(len(method_labels))], args.crop_padding)
                    for col_idx in range(len(method_labels)):
                        panel = panel_data[(case_idx, col_idx, view_idx)]
                        panel.score, panel.mask = apply_crop(panel.score, panel.mask, bounds)
                else:
                    for col_idx in range(len(method_labels)):
                        panel = panel_data[(case_idx, col_idx, view_idx)]
                        bounds = crop_bounds_from_masks([panel.mask], args.crop_padding)
                        panel.score, panel.mask = apply_crop(panel.score, panel.mask, bounds)

    if not args.no_square_panels:
        for panel in panel_data.values():
            panel.score, panel.mask = pad_to_square(panel.score, panel.mask)

    n_rows = len(cases) * len(view_specs)
    n_cols = len(method_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(2.0 * n_cols, 5.5), max(1.45 * n_rows, 3.0)), squeeze=False)
    for col_idx, label in enumerate(method_labels):
        axes[0][col_idx].set_title(label, fontsize=args.title_fontsize, pad=4)
    for case_idx, case in enumerate(cases):
        for col_idx, _label in enumerate(method_labels):
            for view_idx, view in enumerate(view_specs):
                row_idx = case_idx * len(view_specs) + view_idx
                ax = axes[row_idx][col_idx]
                panel = panel_data[(case_idx, col_idx, view_idx)]
                render_score_panel(ax, panel.score, panel.mask, ground_truth=panel.is_ground_truth, presence_cmap=presence_cmap, background_mode=args.background_mode, ground_truth_background_mode=args.ground_truth_background_mode, ground_truth_style=args.ground_truth_style, panel_frame=args.panel_frame)
                if col_idx == 0:
                    ax.set_ylabel(f"{case}\n{view.label}" if view_idx == 0 else view.label, fontsize=args.label_fontsize, rotation=90, labelpad=8)

    if args.add_colorbar or args.show_colorbar:
        add_outside_colorbar(fig, axes, presence_cmap, args)
    else:
        fig.tight_layout()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[OK] Saved presence-score figure: {args.out_path}")


if __name__ == "__main__":
    main()



# simple
'''
python scripts/analysis/plot_presence_score_maps.py \
  --manifest analysis/revise/presence_frequency_maps/manifest_simple_ABC.csv \
  --out_path analysis/revise/presence_frequency_maps/simple_presence_frequency_maps.pdf \
  --case_filter Object_A,Object_B,Object_C \
  --target_color simple_blue \
  --side_length 16 \
  --episode 0 \
  --step 0 \
  --presence_cmap jet_bright \
  --background_mode low_score \
  --ground_truth_style target_only \
  --show_colorbar
'''
