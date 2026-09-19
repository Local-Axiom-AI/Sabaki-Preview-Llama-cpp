# Sabaki runtime notes

This fork adds native inference for the Sabaki v3 GGUF architecture.

## Implemented model features

- 16 pre-norm decoder blocks with ordinary residual additions;
- 12 KDA recurrent sequence mixers with persistent recurrent state;
- 4 global MLA blocks with adjacent-pair RoPE over 256 latent coordinates;
- one dense SiTU-GLU feed-forward block;
- 15 sparse Top-2 MoE blocks with 8 routed and 2 shared experts;
- tied token embeddings and language-model output weights;
- lossless byte-level BPE and Sabaki control tokens;
- CPU and CUDA SiTU operators;
- CUDA expert dispatch and recurrent inference paths.

Sabaki GGUF files use architecture name `sabaki` and architecture version `3`. Earlier experimental weights are incompatible.

## Conversion

From the repository root:

```bash
python tools/convert_sabaki_to_gguf.py \
  /path/to/sabaki-preview.pt \
  --tokenizer /path/to/sabaki-tokenizer.json \
  --outfile /path/to/Sabaki-Preview-F16.gguf
```

The converter validates the canonical v3 dimensions, writes matrix weights in F16, and retains normalization scales, biases, and recurrent convolution kernels in F32. It reverses KDA convolution taps during export because the reference checkpoint numbers the current-token tap first while `ggml_ssm_conv` reads its window from oldest to current.

## Serving

After building the server, supply the GGUF explicitly:

```bash
./build/bin/llama-server \
  --model /path/to/Sabaki-Preview-F16.gguf \
  --ctx-size 16384 \
  --n-gpu-layers 999 \
  --host 127.0.0.1 \
  --port 8080
```

Use a smaller context allocation when memory is limited. Omit GPU-layer offload on CPU-only builds.

## Compatibility

Sabaki uses custom GGUF tensors and runtime operators. Stock llama.cpp does not currently load these files. GGUF files created for earlier experimental architecture versions are rejected rather than interpreted as v3 weights.
