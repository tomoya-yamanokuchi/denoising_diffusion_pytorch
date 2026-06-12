"""Plot voxel-level target-part presence frequency maps from raw predictions.

This is the Fig.10-style visualization. It reads raw_pred_image ensembles via a
manifest and visualizes the fraction of generated samples in which each voxel is
classified as the target part.
"""

from scripts.analysis.plot_presence_score_maps import main


if __name__ == "__main__":
    main()
