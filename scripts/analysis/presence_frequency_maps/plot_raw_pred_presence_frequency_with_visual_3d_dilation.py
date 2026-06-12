"""Plot presence frequency maps with visualization-only 3D dilation.

This is not the clip_ucb_raw axis-wise decision rule. It applies a voxel-wise
maximum filter to a raw-prediction presence-frequency volume for visualizing how
nearby target-part probability regions expand in 3D.
"""

from scripts.analysis.plot_safety_margin_presence_maps import main


if __name__ == "__main__":
    main()
