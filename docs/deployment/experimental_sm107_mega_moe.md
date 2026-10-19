# Experimental FlashInfer SM107 MegaMoE

The `sm107_deep_gemm_mega_moe_experimental` backend runs the FlashInfer CuTe DSL
port of Rubin MegaMoE for DeepSeek-V4.1 and its DSpark draft model. It keeps the
native vLLM routing and input staging, fuses the shared expert, and places the
routed MXFP4 weights in the two GPU locality domains. Shared MXFP8 weights and
scales retain the native layout. Unsupported hardware, geometry, or EPLB is an
error; this explicit backend does not fall back to the C++ MegaMoE.

Start with the ARM64 CUDA 13.4 nightly at vLLM commit
`1b2a666391f2a24228a1436b64d547c34cc6b6ce`. It supplies CuTe DSL 4.8.0,
TVM-FFI 0.1.11, and cuda-python 13.4.1. Install the pinned FlashInfer fork into
that image's Python environment:

```bash
bash tools/install_flashinfer_sm107.sh
```

The script installs FlashInfer revision
`d172bf90d3ff8bc69ea5216211630de84cb64e0f` without resolving or replacing the
image's dependencies. It builds a Python wheel; it does not rebuild vLLM,
DeepGEMM, or FlashMLA. Keep those published native libraries when installing
this branch's Python files.

The installer also fixes the nightly's Quack argument-converter shim to forward
CuTe DSL 4.8 arguments and preserve dataclass constant values. This allows
MegaMoE compilation after Quack has been imported during model warmup.

Select the backend with:

```bash
--kernel-config '{"moe_backend":"sm107_deep_gemm_mega_moe_experimental"}'
```

Die-local routed weights are part of this backend and need no environment flag.
Each target and draft geometry allocates its symmetric buffer and compiles all
BLOCK_M variants during warmup, before CUDA graph capture. Checkpoint reloads
write into the existing registered parameters, preserving graph addresses.
