# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer Sm107 block-scaled dense GEMM with an MXFP8 tensor entry point.

Imported lazily by ``rubin.py`` (it pulls in the CuTe DSL). Kept at module level
because ``cute.jit`` re-executes the decorated function against the module
globals, so ``cute``/``cutlass`` must be module-level names here.
"""

from typing import Optional, Union

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.arch import griddepcontrol_wait
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm107 import (
    Sm107BlockScaledPersistentDenseGemmKernel,
)


class _PdlLaunch:
    """Adds programmatic-dependent-launch to the parent's ``.launch(...)``."""

    def __init__(self, inner):
        self._inner = inner

    def launch(self, *args, **kwargs):
        kwargs["use_pdl"] = True
        return self._inner.launch(*args, **kwargs)


class Sm107Mxfp8GemmKernel(Sm107BlockScaledPersistentDenseGemmKernel):
    """The upstream ``wrapper`` recasts packed-uint8 FP4 storage (k = 2 x bytes);
    MXFP8 operands are already Float8E4M3FN tensors with k = columns.

    ``enable_pdl`` launches with programmatic dependent launch and executes
    griddepcontrol.wait before any work (the upstream Sm107 kernel has no PDL;
    FlashInfer's SM100 kernel waits after its prologue). Completion is the
    implicit trigger for dependents, as in the SM100 kernel.
    """

    def __init__(self, *args, enable_pdl: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_enable_pdl = enable_pdl

    def kernel(self, *args):
        if self.vllm_enable_pdl:
            return _PdlLaunch(self.kernel_pdl(*args))
        return super().kernel(*args)

    @cute.kernel
    def kernel_pdl(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
    ):
        griddepcontrol_wait()
        self.kernel_impl(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            tma_atom_sfa,
            mSFA_mkl,
            tma_atom_sfb,
            mSFB_nkl,
            tma_atom_c,
            mC_mnl,
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            tCtSFA_layout,
            tCtSFB_layout,
            c_smem_layout_staged,
            epi_tile,
            tile_sched_params,
            epilogue_op,
            self.cluster_shape_mn,
            self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1,
            self.is_a_mcast,
            self.is_b_mcast,
            alpha,
        )

    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,  # noqa: E741
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        m = cute.size(mA, mode=[0])
        k = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])
        self.wrapper_ptrs(
            m,
            n,
            k,
            sf_m,
            sf_n,
            sf_k,
            l,
            mA.iterator,
            mB.iterator,
            a_sf_ptr,
            b_sf_ptr,
            mC.iterator,
            alpha_tensor,
            cutlass.Int64(0),
            max_active_clusters,
            current_stream,
            swap_ab,
            epilogue_op,
        )
