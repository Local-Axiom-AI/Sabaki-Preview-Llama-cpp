# Sabaki Preview llama.cpp

This repository is the official [Local-Axiom-AI](https://github.com/Local-Axiom-AI) llama.cpp fork for running **Sabaki Preview**. It adds native CPU and CUDA inference for Sabaki's custom hybrid recurrent/attention Mixture-of-Experts architecture.

This is preview software for a preview model. Interfaces and model metadata may change in later releases.

## Architecture support

The runtime implements the `sabaki` GGUF architecture:

| Component | Sabaki Preview |
| --- | --- |
| Model width and depth | 768, 16 decoder blocks |
| Sequence mixers | 12 KDA recurrent layers and 4 rotary MLA layers |
| Global attention | 12 heads, 256-dimensional MLA latent space |
| Sparse experts | 8 routed experts, Top-2 selection |
| Shared experts | 2 per MoE block |
| Feed-forward activation | SiTU-GLU |
| Vocabulary | 49,152-token lossless byte-level BPE |
| Maximum configured context | 16,384 tokens |

The fork includes Sabaki-specific GGUF metadata, tokenizer handling, recurrent cache behavior, fused SiTU operators, expert dispatch, and model conversion. Stock llama.cpp does not currently load Sabaki GGUF files.

## Clone

```bash
git clone https://github.com/Local-Axiom-AI/Sabaki-Preview-Llama-cpp.git
cd Sabaki-Preview-Llama-cpp
```

## Build

### Linux

CPU:

```bash
cmake -B build
cmake --build build --config Release -j
```

CUDA:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

### Windows

CPU:

```powershell
cmake -B build
cmake --build build --config Release
```

CUDA:

```powershell
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

For additional platforms and build options, see [the upstream build guide](docs/build.md).

## Run Sabaki Preview

Start the OpenAI-compatible server:

```bash
./build/bin/llama-server \
  --model /path/to/Sabaki-Preview-F16.gguf \
  --ctx-size 16384 \
  --n-gpu-layers 999 \
  --host 127.0.0.1 \
  --port 8080
```

On a CPU-only system, omit `--n-gpu-layers 999`. On Windows with a multi-config generator, the executable may be under `build/bin/Release`.

Run the command-line client:

```bash
./build/bin/llama-cli \
  --model /path/to/Sabaki-Preview-F16.gguf \
  --ctx-size 16384
```

The configured 16K context is an architectural limit, not a claim of validated 16K recall quality.

## Convert a checkpoint

The included converter produces a Sabaki FP16 GGUF from a compatible v3 reference checkpoint and tokenizer:

```bash
python tools/convert_sabaki_to_gguf.py \
  /path/to/sabaki-preview.pt \
  --tokenizer /path/to/sabaki-tokenizer.json \
  --outfile /path/to/Sabaki-Preview-F16.gguf
```

The converter requires Python, PyTorch, and NumPy. See [Sabaki runtime notes](docs/sabaki.md) for model-specific details.

## Project status

- This fork targets Sabaki Preview and is maintained separately from upstream llama.cpp.
- Only Sabaki-specific inference and conversion paths are supported by Local-Axiom-AI.
- General llama.cpp usage and platform documentation remain available in the inherited `docs` directory.
- Model weights are distributed separately under their own license.

## Upstream project

This project is based on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The original project and contributors retain their respective copyrights. Sabaki support is maintained by Local-Axiom-AI and is not an upstream llama.cpp feature.

## License

The source code in this repository is distributed under the [MIT License](LICENSE), consistent with upstream llama.cpp. The Sabaki model weights are separate artifacts and are not covered by this repository's MIT license.
