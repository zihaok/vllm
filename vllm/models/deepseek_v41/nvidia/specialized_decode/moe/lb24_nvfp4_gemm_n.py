# ruff: noqa: E501
"""The NVFP4 low-batch GEMM1 for 9-24 rows.

RoutedExpertGemm1N (lb24_kernels.py: the executor's persistent GEMM1 with a 16 / 24 column B tile, the accumulator read 8 columns
at a time) at NVFP4 w4a4, the same way lb24_nvfp4_gemm.NvGemm1 ports the executor's 8-row GEMM1:
* sf_vec_size 16, E4M3 scales, E2M1 operands (kind mxf4nvf4);
* x's scales: the B load warp writes them from x's linear E4M3 scales (rows, hidden / 16), four codes per 32-bit word, into rows
  0 .. cta_n - 1 of B's scale atoms (one 512-byte atom per 64 columns at 16; rows past x's end get scale 1.0);
* per 8-column group: alpha1[e] x accumulator, clamped SwiGLU, NVFP4 H with the amax of a 16-column block inside one warp
  (lb24_nvfp4_gemm's quantizer), so no exchange and no named barrier; padding columns zero with scale 1.0.
GEMM2 is lb24_nvfp4_gemm.NvGemm2 (it takes any number of 8-row tiles). The route kernel is lb24_kernels.RoutedExpertRouteN.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32, Int64, Uint8, Uint16, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineUmmaAsync,
    PipelineUserType,
    make_pipeline_state,
    pipeline_init_arrive,
    pipeline_init_wait,
)

from . import lb24_kernels as L
from . import lb24_nvfp4_gemm as NV

G = NV.G
TILE = NV.TILE


class NvGemm1N(L.RoutedExpertGemm1N):
    """GEMM1 at NVFP4 for up to ``cta_n`` rows (module docstring)."""

    def __init__(
        self,
        hidden: int,
        slice_: int,
        padded: int,
        cta_n: int = 16,
        cta_k: int = 128,
        num_ab_stage: int = 11,
    ) -> None:
        super().__init__(hidden, slice_, padded, cta_n, cta_k, num_ab_stage)
        self.sf_vec_size = NV.NV_SF_VEC

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        b_scale_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        hs_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        clamp_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        h_rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w13 (2 x padded, hidden, experts) E2M1 with E4M3 scales (128x4 layout); B: x (rows, hidden) E2M1, ``b_scale_ptr``
        its linear E4M3 scales as Uint32 words (rows, hidden / 64); H (h_rows, padded / 2) bytes; HS per-tile at 16."""
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((self.m, self.k, experts), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr,
            NV.blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size),
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((rows, self.k, 1), order=(1, 0, 2))
        )
        words = self.k // 64
        b_scale = cute.make_tensor(
            b_scale_ptr, cute.make_layout((rows, words), stride=(words, 1))
        )
        h = cute.make_tensor(h_ptr, cute.make_layout(h_rows * (self.padded // 2)))
        hs = cute.make_tensor(
            hs_ptr,
            cute.make_layout(
                (h_rows + TILE - 1) // TILE * NV.nv_tile_scale_bytes(self.padded)
            ),
        )
        clamp = cute.make_tensor(clamp_ptr, cute.make_layout(experts))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(experts))
        gs = cute.make_tensor(gs_ptr, cute.make_layout(experts))
        route = cute.make_tensor(
            route_ptr, cute.make_layout(4 + 2 * cta_words + h_rows)
        )
        self._launch(
            a,
            sfa,
            b,
            None,
            route,
            (h, hs, clamp, alpha, gs),
            (b_scale,),
            cta_words,
            ctas,
            stream,
        )

    @cute.jit
    def write_b_scales(
        self, sSFB: cute.Tensor, stage: Int32, k_tile: Int32, b_side: tuple
    ):
        """Rows 0 .. cta_n - 1 of B's scale atoms for one stage (one 512-byte atom per 64 columns; rows past x's end 1.0)."""
        (b_scale,) = b_side
        atoms = self.cta_k // 64
        stage_bytes = 512 * atoms
        words = cute.make_tensor(
            cute.recast_ptr(sSFB.iterator, dtype=Uint32),
            cute.make_layout(self.num_ab_stage * stage_bytes // 4),
        )
        lane = cute.arch.lane_idx()
        for idx in cutlass.range_constexpr(0, self.cta_n * atoms, 32):
            entry = lane + idx
            if entry < self.cta_n * atoms:
                row = entry // atoms
                atom = entry - row * atoms
                value = Uint32(0x38383838)
                if row < cute.size(b_scale, mode=[0]):
                    value = Uint32(b_scale[row, k_tile * atoms + atom])
                words[(stage * stage_bytes + atom * 512 + row * 16) // 4] = value
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()

    @cute.jit
    def store_group(
        self,
        tDrAcc: cute.Tensor,
        group: cutlass.Constexpr[int],
        epi: tuple,
        mRoute: cute.Tensor,
        cta_words: Int32,
        n_tile: Int32,
        m_tile: Int32,
        item: Int32,
        exchange: cute.Tensor,
    ):
        """lb24_nvfp4_gemm.NvGemm1.store for accumulator columns 8 group .. 8 group + 7 (tokens 8 group + 2 (lane % 4) + {0, 1});
        skipped when no row of the tile holds a token of the group."""
        mH, mHS, mClamp, mAlpha, mGs = epi
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - self.epilogue_warp
        lane = cute.arch.lane_idx()
        token = (lane % 4) * 2 + TILE * group
        pair = warp * 16 + (lane // 4) * 2
        expert = mRoute[4 + n_tile]
        limit = mClamp[expert]
        alpha = mAlpha[expert]
        gs = mGs[expert]
        row0 = n_tile * TILE
        valid = mRoute[4 + cta_words + n_tile] - row0
        row_a = Int32(-1)
        row_b = Int32(-1)
        hit = Int32(0)
        for r in cutlass.range_constexpr(TILE):
            if r < valid:
                routed = mRoute[4 + 2 * cta_words + row0 + r]
                if routed >= TILE * group and routed < TILE * (group + 1):
                    hit = Int32(1)
                if routed == token:
                    row_a = Int32(r)
                if routed == token + 1:
                    row_b = Int32(r)
        if hit != 0:
            lo_a, lo_b = G.clamped_swiglu_x2(
                tDrAcc[0] * alpha,
                tDrAcc[1] * alpha,
                tDrAcc[2] * alpha,
                tDrAcc[3] * alpha,
                limit,
                -limit,
            )
            hi_a, hi_b = G.clamped_swiglu_x2(
                tDrAcc[4] * alpha,
                tDrAcc[5] * alpha,
                tDrAcc[6] * alpha,
                tDrAcc[7] * alpha,
                limit,
                -limit,
            )
            amax_a = G.abs_max_x2(lo_a, hi_a)
            amax_b = G.abs_max_x2(lo_b, hi_b)
            for shift in cutlass.range_constexpr(4, 1, -1):
                amax_a = cute.arch.fmax(
                    amax_a, cute.arch.shuffle_sync_bfly(amax_a, 1 << shift), ftz=True
                )
                amax_b = cute.arch.fmax(
                    amax_b, cute.arch.shuffle_sync_bfly(amax_b, 1 << shift), ftz=True
                )
            values, codes = NV.nvfp4_x2(amax_a, amax_b, lo_a, hi_a, lo_b, hi_b, gs)
            h_row = self.padded // 2
            column = (m_tile * 64 + pair) // 2
            if row_a >= 0:
                mH[(row0 + row_a) * h_row + column] = Uint8(values & Uint16(0xFF))
            if row_b >= 0:
                mH[(row0 + row_b) * h_row + column] = Uint8(values >> Uint16(8))
            if lane < 4:
                base = (
                    n_tile * NV.nv_tile_scale_bytes(self.padded) + m_tile * 512 + warp
                )
                if row_a >= 0:
                    mHS[base + row_a * 16] = Uint8(codes & Uint16(0xFF))
                if row_b >= 0:
                    mHS[base + row_b * 16] = Uint8(codes >> Uint16(8))

    @cute.jit
    def store_padding(
        self,
        epi: tuple,
        mRoute: cute.Tensor,
        cta_words: Int32,
        n_tile: Int32,
        m_tile: Int32,
    ):
        """H's padding columns of the tile's rows: zero values, scale 1.0 (E4M3 0x38)."""
        mH, mHS, mClamp, mAlpha, mGs = epi
        if cutlass.const_expr(self.padded > self.slice):  # noqa: SIM102
            if m_tile == self.m_tiles - 1:
                warp = (
                    cute.arch.make_warp_uniform(cute.arch.warp_idx())
                    - self.epilogue_warp
                )
                lane = cute.arch.lane_idx()
                row0 = n_tile * TILE
                valid = mRoute[4 + cta_words + n_tile] - row0
                h_row = self.padded // 2
                tile_bytes = NV.nv_tile_scale_bytes(self.padded)
                thread = warp * 32 + lane
                pad_row = thread // 16
                if pad_row < valid:
                    pad = (row0 + pad_row) * h_row + self.slice // 2 + (thread % 16) * 2
                    mH[pad] = Uint8(0)
                    mH[pad + 1] = Uint8(0)
                if thread < 4 * TILE:
                    pad_row = thread // 4
                    pad_block = self.slice // 16 + thread % 4
                    if pad_row < valid:
                        mHS[
                            n_tile * tile_bytes
                            + (pad_block // 4) * 512
                            + pad_row * 16
                            + pad_block % 4
                        ] = Uint8(NV.E4M3_ONE)


class NvGemm1NT(NvGemm1N):
    """NvGemm1N with x's E4M3 scales loaded by TMA from one 128-row atom in the 128x4 layout (as NvGemm1 does at <= 8 rows), in
    place of the B warp's per-stage global loads (measured: NvGemm1N was 24-49 % slower at 18 / 24 rows; that warp issued 40 x 1-2
    dependent global loads per item at cta_k 128)."""

    def __init__(
        self,
        hidden: int,
        slice_: int,
        padded: int,
        cta_n: int = 16,
        cta_k: int = 256,
        num_ab_stage: int = 8,
    ) -> None:
        super().__init__(hidden, slice_, padded, cta_n, cta_k, num_ab_stage)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        hs_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        clamp_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        h_rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """As NvGemm1N, with ``sfb_ptr`` x's E4M3 scales as one 128-row atom set in the 128x4 layout (128 x hidden / 16 bytes)."""
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((self.m, self.k, experts), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr,
            NV.blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size),
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((rows, self.k, 1), order=(1, 0, 2))
        )
        sfb = cute.make_tensor(
            sfb_ptr,
            NV.blockscaled_utils.tile_atom_to_shape_SF(
                cute.make_ordered_layout((128, self.k, 1), order=(1, 0, 2)).shape,
                self.sf_vec_size,
            ),
        )
        h = cute.make_tensor(h_ptr, cute.make_layout(h_rows * (self.padded // 2)))
        hs = cute.make_tensor(
            hs_ptr,
            cute.make_layout(
                (h_rows + TILE - 1) // TILE * NV.nv_tile_scale_bytes(self.padded)
            ),
        )
        clamp = cute.make_tensor(clamp_ptr, cute.make_layout(experts))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(experts))
        gs = cute.make_tensor(gs_ptr, cute.make_layout(experts))
        route = cute.make_tensor(
            route_ptr, cute.make_layout(4 + 2 * cta_words + h_rows)
        )
        self._launch(
            a,
            sfa,
            b,
            sfb,
            route,
            (h, hs, clamp, alpha, gs),
            (),
            cta_words,
            ctas,
            stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfb: Optional[cute.CopyAtom],  # noqa: UP045
        mSFB_nkl: Optional[cute.Tensor],  # noqa: UP045
        mRoute: cute.Tensor,
        epi: tuple,
        b_side: tuple,
        cta_words: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
    ):
        """RoutedExpertGemm1N.kernel (lb24_kernels.py) with the B load warp of the executor's 8-row kernel:
        x and its scales both by TMA, so the warp issues two copies per stage and no global load."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        mma_tile_coord_v = Int32(0)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        AB_Stage = self.num_ab_stage
        Acc_Stage = self.num_acc_stage
        EXCHANGE = 2 * self.groups * 4 * TILE

        @cute.struct
        class SharedStorage:
            ab_pipeline_bars: cute.struct.MemRange[cutlass.Int64, AB_Stage * 2]
            acc_pipeline_bars: cute.struct.MemRange[cutlass.Int64, Acc_Stage * 2]
            tmem_barrier: cutlass.Int64
            tmem_base_ptr: cutlass.Int32
            # GEMM1's block amax exchange: one buffer per (item parity, column group).
            epi_exchange: cute.struct.MemRange[cutlass.Float32, EXCHANGE]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_pipeline_bars.data_ptr(),
            num_stages=AB_Stage,
            producer_group=CooperativeGroup(Agent.Thread, 2),
            consumer_group=CooperativeGroup(Agent.Thread),
            defer_sync=True,
        )
        ab_full_bar = ab_pipeline.sync_object_full.barrier_storage
        ab_empty_bar = ab_pipeline.sync_object_empty.barrier_storage
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_pipeline_bars.data_ptr(),
            num_stages=Acc_Stage,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, 128),
            defer_sync=True,
        )
        tmem_bar = storage.tmem_barrier.ptr
        tmem_base_smem_ptr = storage.tmem_base_ptr.ptr
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_bar, 32 + 128)
        pipeline_init_arrive(cluster_shape_mn=None, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=None)

        sA = smem.allocate_tensor(
            element_type=self.smem_alloc_a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.smem_alloc_b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        # GEMM1 reads the record only after its grid dependency wait (route_before_wait False).
        cute.arch.griddepcontrol_wait()
        total = mRoute[1] * self.m_tiles
        count = Int32(0)
        if bid < total:
            count = (total - 1 - bid) // grid + 1

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        if warp_idx == 0:
            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            gA_mkl = cute.local_tile(
                mA_mkl, (self.cta_m, self.cta_k), (None, None, None)
            )
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gA_mkl), 0, 3),
            )
            gSFA_mkl = cute.local_tile(
                mSFA_mkl, (self.cta_m, self.cta_k), (None, None, None)
            )
            tAsSFA, tAgSFA = cpasync.tma_partition(
                tma_atom_sfa,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sSFA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gSFA_mkl), 0, 3),
            )
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)
            policy = Int64(G.EVICT_FIRST)
            state = make_pipeline_state(PipelineUserType.Producer, AB_Stage)
            for i in cutlass.range(count, unroll=1):
                item = bid + i * grid
                n_tile = item // self.m_tiles
                m_tile = item - n_tile * self.m_tiles
                expert = mRoute[4 + n_tile]
                tAgA_item = tAgA[(None, m_tile, None, expert)]
                tAgSFA_item = tAgSFA[(None, m_tile, None, expert)]
                for k_tile in cutlass.range(self.k_tiles, unroll=1):
                    cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            ab_full_bar + state.index, self.tma_bytes_a
                        )
                    cute.copy(
                        tma_atom_a,
                        tAgA_item[(None, k_tile)],
                        tAsA[(None, state.index)],
                        tma_bar_ptr=ab_full_bar + state.index,
                        cache_policy=policy,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_item[(None, k_tile)],
                        tAsSFA[(None, state.index)],
                        tma_bar_ptr=ab_full_bar + state.index,
                        cache_policy=policy,
                    )
                    state.advance()
            cute.arch.griddepcontrol_launch_dependents()
            for _ in cutlass.range(AB_Stage, unroll=1):
                cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                state.advance()
        elif warp_idx == 1:
            # x and its E4M3 scales by TMA (the 8-row NvGemm1's B side; RoutedExpertGemm1N's warp wrote the scales itself).
            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            gB_nkl = cute.local_tile(
                mB_nkl, (self.cta_n, self.cta_k), (None, None, None)
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(thr_mma.partition_B(gB_nkl), 0, 3),
            )
            gSFB_nkl = cute.local_tile(mSFB_nkl, (128, self.cta_k), (None, None, None))
            tBsSFB, tBgSFB = cpasync.tma_partition(
                tma_atom_sfb,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sSFB, 0, 3),
                cute.group_modes(
                    tiled_mma_sfb.get_slice(mma_tile_coord_v).partition_B(gSFB_nkl),
                    0,
                    3,
                ),
            )
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)
            state = make_pipeline_state(PipelineUserType.Producer, AB_Stage)
            tBgB_item = tBgB[(None, 0, None, 0)]
            tBgSFB_item = tBgSFB[(None, 0, None, 0)]
            for i in cutlass.range(count, unroll=1):
                for k_tile in cutlass.range(self.k_tiles, unroll=1):
                    cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            ab_full_bar + state.index, self.tma_bytes_b
                        )
                    cute.copy(
                        tma_atom_b,
                        tBgB_item[(None, k_tile)],
                        tBsB[(None, state.index)],
                        tma_bar_ptr=ab_full_bar + state.index,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_item[(None, k_tile)],
                        tBsSFB[(None, state.index)],
                        tma_bar_ptr=ab_full_bar + state.index,
                    )
                    state.advance()
            for _ in cutlass.range(AB_Stage, unroll=1):
                cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                state.advance()
        elif warp_idx == 2:
            self.mma_warp(
                ab_full_bar,
                ab_empty_bar,
                acc_pipeline,
                tmem_bar,
                tmem_base_smem_ptr,
                tiled_mma,
                sA,
                sB,
                sSFA,
                sSFB,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                count,
            )
        elif warp_idx >= self.epilogue_warp:
            epi_tid = tidx - 32 * self.epilogue_warp
            cute.arch.mbarrier_arrive(tmem_bar)
            cute.arch.mbarrier_wait(tmem_bar, 0)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, 16, tmem_base_smem_ptr
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, self._acc_layout(tiled_mma))
            # The 8-column kernel's accumulator view, TMEM load and register shape, applied to each 8-column group.
            tiled_mma8 = self._tiled_mma((self.cta_m, TILE))
            acc8_layout = tiled_mma8.make_fragment_C(
                tiled_mma8.partition_shape_C((self.cta_m, TILE))
            ).layout
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                (self.cta_m, TILE, self.cta_k),
                cutlass.utils.LayoutEnum.COL_MAJOR,
                cutlass.BFloat16,
                self.acc_dtype,
                (self.cta_m, TILE),
                False,
            )
            acc8_ref = cute.make_tensor(tmem_ptr, acc8_layout)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, acc8_ref[((None, None), 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)
            thr_mma8 = tiled_mma8.get_slice(mma_tile_coord_v)
            coord_c = cute.make_identity_tensor((self.m, TILE, 1))
            tCcC = thr_mma8.partition_C(
                cute.local_tile(coord_c, (self.cta_m, TILE), (None, None, None))
            )
            tDcC_all = thr_copy_t2r.partition_D(
                tCcC[((None, None), 0, 0, None, None, None)]
            )
            exchange = cute.make_tensor(
                storage.epi_exchange.data_ptr(), cute.make_layout(EXCHANGE)
            )
            state = make_pipeline_state(PipelineUserType.Consumer, Acc_Stage)
            for i in cutlass.range(count, unroll=1):
                item = bid + i * grid
                n_tile = item // self.m_tiles
                m_tile = item - n_tile * self.m_tiles
                acc_pipeline.consumer_wait(state)
                stage_ptr = tCtAcc_base[(None, None, None, state.index)].iterator
                regs = []
                for j in cutlass.range_constexpr(self.groups):
                    acc_j = cute.make_tensor(stage_ptr + TILE * j, acc8_layout)
                    src = thr_copy_t2r.partition_S(acc_j[((None, None), 0, 0)])
                    reg = cute.make_rmem_tensor(
                        tDcC_all[(None, None, None, 0, 0, 0)].shape, self.acc_dtype
                    )
                    cute.copy(tiled_copy_t2r, src, reg)
                    regs.append(reg)
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(state)
                state.advance()
                for j in cutlass.range_constexpr(self.groups):
                    self.store_group(
                        regs[j], j, epi, mRoute, cta_words, n_tile, m_tile, i, exchange
                    )
                self.store_padding(epi, mRoute, cta_words, n_tile, m_tile)
            cute.arch.mbarrier_arrive(tmem_bar)
            cute.arch.mbarrier_wait(tmem_bar, 1)
        cute.arch.barrier()
