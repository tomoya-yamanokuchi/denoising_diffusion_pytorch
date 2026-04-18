# Lazy imports — only load modules when explicitly accessed.
# This prevents import chain failures from optional dependencies
# (scikit-image, tap, pyvista, etc.) breaking the entire utils package.

from .arrays import *
from .make_save_path import make_save_path

# The following modules have external dependencies and are imported
# on demand by the code that needs them:
#   .serialization  — needs vaeac_utils (skimage)
#   .setup          — needs tap
#   .config         — needs omegaconf
#   .progress       — standalone but not needed at init
