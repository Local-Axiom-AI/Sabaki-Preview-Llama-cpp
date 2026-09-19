#!/usr/bin/env python3
"""Convert a canonical Sabaki PyTorch checkpoint to this fork's GGUF format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402
from gguf.vocab import bytes_to_unicode  # noqa: E402


CONTROL_TOKENS = (
    "<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|system_end|>",
    "<|developer|>", "<|developer_end|>", "<|user|>", "<|user_end|>",
    "<|assistant|>", "<|assistant_end|>", "<|tool_schema|>", "<|tool_schema_end|>",
    "<|tool_call|>", "<|tool_call_end|>", "<|parallel_tool_calls|>",
    "<|parallel_tool_calls_end|>", "<|tool_result|>", "<|tool_result_end|>",
    "<|tool_error|>", "<|tool_error_end|>", "<|final|>", "<|final_end|>",
    "<|memory|>", "<|memory_end|>", "<|confirmation_required|>",
    "<|confirmation_required_end|>", "<|confirmation_granted|>",
    "<|confirmation_denied|>", "<|citation|>", "<|citation_end|>",
    "<|observation|>", "<|observation_end|>", "<|sep|>", "<|json|>",
    "<|json_end|>", "<|cancelled|>", "<|truncated|>", "<|unknown|>",
    "<|tool_denied|>", "<|tool_denied_end|>",
)

MLA_LAYERS = {3, 7, 11, 15}


def as_f16(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(dtype=torch.float16, device="cpu").contiguous().numpy()


def as_f32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()


def add_tokenizer(writer: gguf.GGUFWriter, tokenizer_path: Path) -> None:
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if tuple(data["control_tokens"]) != CONTROL_TOKENS or data["vocab_size"] != 49152:
        raise ValueError("tokenizer is not the canonical Sabaki tokenizer")

    merges = [tuple(pair) for pair in data["merges"]]
    byte_encoder = bytes_to_unicode()
    symbols: dict[int, str] = {256 + b: byte_encoder[b] for b in range(256)}
    tokens = list(CONTROL_TOKENS)
    token_types = [gguf.TokenType.CONTROL] * len(tokens)
    for i in range(len(tokens), 256):
        tokens.append(f"[Sabaki unused {i}]")
        token_types.append(gguf.TokenType.UNUSED)
    for b in range(256):
        tokens.append(symbols[256 + b])
        token_types.append(gguf.TokenType.NORMAL)

    merge_text: list[str] = []
    for rank, (left, right) in enumerate(merges):
        if left not in symbols or right not in symbols:
            raise ValueError(f"merge {rank} refers to an unknown token")
        merge_text.append(f"{symbols[left]} {symbols[right]}")
        symbols[512 + rank] = symbols[left] + symbols[right]
        tokens.append(symbols[512 + rank])
        token_types.append(gguf.TokenType.NORMAL)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("sabaki")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merge_text)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(0)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    writer.add_chat_template(
        "{% if messages|length > 0 %}<|bos|>"
        "{% if messages[0]['role'] != 'system' %}"
        "<|system|>You are Sabaki, a helpful local assistant. No external tools are available. Answer directly.<|system_end|>"
        "{% endif %}{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|system|>{{ message['content'] }}<|system_end|>"
        "{% elif message['role'] == 'developer' %}<|developer|>{{ message['content'] }}<|developer_end|>"
        "{% elif message['role'] == 'user' %}<|user|>{{ message['content'] }}<|user_end|>"
        "{% elif message['role'] == 'assistant' %}<|assistant|><|final|>{{ message['content'] }}<|final_end|><|assistant_end|>"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )


def add_tensor(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor) -> None:
    # Keep normalization scales and additive biases in F32.
    # Besides preserving their precision, llama.cpp's CUDA broadcast kernels
    # expect these vectors/biases to match the F32 activation accumulator.
    use_f32 = tensor.ndim == 1 or name.endswith(".bias") or ".ssm_conv1d_" in name
    writer.add_tensor(name, as_f32(tensor) if use_f32 else as_f16(tensor))


def validate_payload(payload: dict) -> dict[str, torch.Tensor]:
    checkpoint_format = payload.get("format")
    if not isinstance(checkpoint_format, str) or not checkpoint_format.endswith("-reference-v3"):
        raise ValueError("v3 export requires a compatible v3 reference checkpoint")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state dictionary")
    expected = {
        "vocab_size": 49152, "d_model": 768, "num_layers": 16,
        "num_heads": 12, "head_dim": 64, "context_length": 16384,
        "conv_kernel_size": 4, "decay_rank": 64, "decay_min_log": -5.0,
        "mla_latent_dim": 256, "dense_hidden_dim": 2048,
        "rope_theta": 10000.0,
        "num_routed_experts": 8, "routed_top_k": 2,
        "routed_latent_dim": 384, "routed_hidden_dim": 512,
        "num_shared_experts": 2, "shared_hidden_dim": 768,
        "rms_norm_eps": 1e-6,
        "situ_gate_cap": 4.0, "situ_up_cap": 25.0,
    }
    config = payload.get("config", {})
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    architecture_version = config.get("architecture_version")
    if not isinstance(architecture_version, str) or not architecture_version.endswith("-v3"):
        mismatches["architecture_version"] = (architecture_version, "v3")
    if mismatches or state["token_embedding.weight"].shape != (49152, 768):
        raise ValueError(f"only canonical Sabaki v3 is supported; mismatches: {mismatches}")
    return state


def convert(checkpoint: Path, tokenizer: Path, output: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = validate_payload(payload)

    writer = gguf.GGUFWriter(output, "sabaki")
    writer.add_name("Sabaki v3 222M")
    writer.add_description("Sabaki Preview llama.cpp runtime model")
    writer.add_file_type(1)  # F16
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_vocab_size(49152)
    writer.add_context_length(16384)
    writer.add_rope_dimension_count(256)
    writer.add_rope_freq_base(10000.0)
    writer.add_embedding_length(768)
    writer.add_block_count(16)
    writer.add_feed_forward_length(2048)
    writer.add_head_count(12)
    writer.add_head_count_kv([1 if i in MLA_LAYERS else 0 for i in range(16)])
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_key_length_mla(256)
    writer.add_value_length_mla(256)
    writer.add_kv_lora_rank(256)
    writer.add_ssm_conv_kernel(4)
    writer.add_kda_head_dim(64)
    writer.add_expert_count(8)
    writer.add_expert_used_count(2)
    writer.add_expert_feed_forward_length(512)
    writer.add_expert_shared_count(2)
    writer.add_expert_shared_feed_forward_length(1536)
    writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    writer.add_moe_latent_size(384)
    writer.add_string("sabaki.variant", "canonical-v3-rope-residual")
    writer.add_uint32("sabaki.architecture_version", 3)
    writer.add_float32("sabaki.decay_min_log", -5.0)
    writer.add_float32("sabaki.situ_gate_cap", 4.0)
    writer.add_float32("sabaki.situ_up_cap", 25.0)
    add_tokenizer(writer, tokenizer)

    add_tensor(writer, "token_embd.weight", state["token_embedding.weight"])
    add_tensor(writer, "output_norm.weight", state["final_norm.weight"])

    for i in range(16):
        p = f"blocks.{i}"
        b = f"blk.{i}"
        add_tensor(writer, f"{b}.attn_norm.weight", state[f"{p}.attn_norm.weight"])
        add_tensor(writer, f"{b}.ffn_norm.weight", state[f"{p}.ffn_norm.weight"])

        a = f"{p}.attn"
        add_tensor(writer, f"{b}.attn_q.weight", state[f"{a}.q_proj.weight"])
        add_tensor(writer, f"{b}.attn_output.weight", state[f"{a}.out_proj.weight"])
        add_tensor(writer, f"{b}.ssm_g_a.weight", state[f"{a}.output_gate.weight"])
        if i in MLA_LAYERS:
            add_tensor(writer, f"{b}.attn_kv_a_mqa.weight", state[f"{a}.kv_down.weight"])
            add_tensor(writer, f"{b}.attn_kv_a_norm.weight", state[f"{a}.kv_norm.weight"])
            add_tensor(writer, f"{b}.attn_k_b.weight", state[f"{a}.k_up"].transpose(1, 2))
            add_tensor(writer, f"{b}.attn_v_b.weight", state[f"{a}.v_up"])
        else:
            add_tensor(writer, f"{b}.attn_k.weight", state[f"{a}.k_proj.weight"])
            add_tensor(writer, f"{b}.attn_v.weight", state[f"{a}.v_proj.weight"])
            for qkv in ("q", "k", "v"):
                # Sabaki stores tap 0 as the current token, while ggml_ssm_conv
                # consumes its window from oldest to current. Reverse the taps.
                conv = state[f"{a}.{qkv}_conv.weight"].flip(-1).reshape(1, 768, 1, 4)
                add_tensor(writer, f"{b}.ssm_conv1d_{qkv}.weight", conv)
            add_tensor(writer, f"{b}.ssm_f_a.weight", state[f"{a}.alpha_down.weight"])
            add_tensor(writer, f"{b}.ssm_f_b.weight", state[f"{a}.alpha_up.weight"])
            add_tensor(writer, f"{b}.ssm_dt.bias", state[f"{a}.alpha_bias"])
            add_tensor(writer, f"{b}.ssm_a", state[f"{a}.alpha_log_scale"])
            add_tensor(writer, f"{b}.ssm_beta.weight", state[f"{a}.beta_proj.weight"])
            add_tensor(writer, f"{b}.ssm_norm.weight", state[f"{a}.head_norm.weight"].flatten())

        f = f"{p}.ffn"
        if i == 0:
            add_tensor(writer, f"{b}.ffn_gate.weight", state[f"{f}.gate_proj.weight"])
            add_tensor(writer, f"{b}.ffn_up.weight", state[f"{f}.up_proj.weight"])
            add_tensor(writer, f"{b}.ffn_down.weight", state[f"{f}.out_proj.weight"])
        else:
            add_tensor(writer, f"{b}.ffn_gate_inp.weight", state[f"{f}.router.weight"])
            add_tensor(writer, f"{b}.exp_probs_b.bias", state[f"{f}.qb_bias"])
            add_tensor(writer, f"{b}.ffn_latent_down.weight", state[f"{f}.latent_down.weight"])
            add_tensor(writer, f"{b}.ffn_norm_exps.weight", state[f"{f}.latent_norm.weight"])
            add_tensor(writer, f"{b}.ffn_latent_up.weight", state[f"{f}.latent_up.weight"])
            for src, dst in (("gate_proj", "gate"), ("up_proj", "up"), ("out_proj", "down")):
                stacked = torch.stack([state[f"{f}.routed.{e}.{src}.weight"] for e in range(8)])
                add_tensor(writer, f"{b}.ffn_{dst}_exps.weight", stacked)
            for src, dst in (("gate_proj", "gate"), ("up_proj", "up")):
                packed = torch.cat([state[f"{f}.shared.{e}.{src}.weight"] for e in range(2)], dim=0)
                add_tensor(writer, f"{b}.ffn_{dst}_shexp.weight", packed)
            packed_down = torch.cat([state[f"{f}.shared.{e}.out_proj.weight"] for e in range(2)], dim=1)
            add_tensor(writer, f"{b}.ffn_down_shexp.weight", packed_down)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    args = parser.parse_args()
    convert(args.checkpoint.resolve(), args.tokenizer.resolve(), args.outfile.resolve())


if __name__ == "__main__":
    main()
