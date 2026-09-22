from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

from denoising_diffusion_pytorch.env.mesh_components import (
    MeshComponent,
    MeshComponentSet,
)
from denoising_diffusion_pytorch.utils.voxel_handlers import (
    pv_box_array_multi_type_obj,
)


def load_precomputed_voxel_cells(cache_path: Path):
    if not cache_path.exists():
        print(f"[WARNING] voxel cache not found: {cache_path}")
        print("[INFO] continue without cache.")
        return None

    print(f"[INFO] loading voxel cache: {cache_path}")
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--body-stl",
        type=Path,
        required=True,
        help="Path to Body.stl",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path",
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=49,
        help="Voxel grid side length",
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional path to my_dict49.pkl etc.",
    )

    parser.add_argument(
        "--body-color",
        type=float,
        nargs=3,
        default=(0.9, 0.9, 0.9),
        help="RGB color for body, range [0,1]",
    )

    args = parser.parse_args()

    body_stl = args.body_stl.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not body_stl.exists():
        raise FileNotFoundError(f"Body STL not found: {body_stl}")

    dim = args.dim
    body_color = tuple(args.body_color)

    if args.cache is None:
        cache_path = Path(f"./my_dict{dim}.pkl")
    else:
        cache_path = args.cache.resolve()

    precomputed_cells = load_precomputed_voxel_cells(cache_path)

    print(f"[INFO] reading mesh: {body_stl}")
    body_mesh = pv.read(str(body_stl))

    components = MeshComponentSet(
        {
            "Body": MeshComponent(
                name="Body",
                mesh=body_mesh,
                color=body_color,
            )
        }
    )

    grid_config = {
        "bounds": (
            -0.3, 0.3,
            -0.3, 0.3,
            -0.3, 0.3,
        ),
        "side_length": dim,
    }

    voxel_handler = pv_box_array_multi_type_obj(
        grid_config=grid_config,
        pre_near_by_cells=precomputed_cells,
    )

    voxel_handler.cast_mesh_to_box_array(mesh_components=components)

    box_arrays_data = voxel_handler.get_box_array_data()
    colors = box_arrays_data.colors

    img = voxel_handler.get_box_color_to_2d_image(
        box_color=colors,
        permute="z",
    )

    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(img_uint8).save(output_path)

    print(f"[SAVE] {output_path}")
    print(f"[INFO] image shape: {img.shape}")


if __name__ == "__main__":
    main()
