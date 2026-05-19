"""QCRegNet-style quasiconformal registration prototype for tactile marker images.

This package is intentionally independent from the existing VoxelMorph code path.
It follows the QCRegNet idea:

    image pair -> Estimator -> Beltrami coefficient mu -> BSNet -> mapping -> warp

The training signal is image similarity between the fixed image and the warped moving image.
"""

from .models import Estimator, BSNet
