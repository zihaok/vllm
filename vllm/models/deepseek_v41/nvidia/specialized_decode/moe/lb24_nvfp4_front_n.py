# ruff: noqa: E501
"""The fused NVFP4 front for 9-24 rows: the 24-row front (lb24_kernels.RoutedExpertFrontN2: the executor's top-k selection with the
(row, part) units looped over CTA 0's 32 warps, every unit's logits loaded before selection, and the multi-tile record) with
lb24_nvfp4_front.FrontNV's NVFP4 quantizing CTAs (three rows per CTA; scales in the 128x4 layout, rows < 32).
No new device code: the class only combines the two (method resolution: FrontNV's launch and quantization, FrontN2's selection)."""

from . import lb24_kernels as LK
from . import lb24_nvfp4_front as FN


class FrontNVN2(FN.FrontNV, LK.RoutedExpertFrontN2):
    """FrontNV's NVFP4 launch and quantizing CTAs with the 24-row selection and record (rows in [1, 24])."""
