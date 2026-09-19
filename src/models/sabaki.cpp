#include "models.h"
#include "llama-memory-recurrent.h"

#include <cmath>

void llama_model_sabaki::load_arch_hparams(llama_model_loader & ml) {
    uint32_t version = 0;
    ml.get_key("sabaki.architecture_version", version, false);
    if (version != 3) {
        throw std::runtime_error("Sabaki v3 requires a v3 RoPE/residual GGUF; legacy weights are incompatible");
    }
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,        hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,           hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,         hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,             hparams.n_lora_kv);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,                    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                       hparams.n_embd_head_kda);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,         hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH,  hparams.n_ff_shexp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,                hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                 hparams.expert_gating_func);
    ml.get_key(LLM_KV_MOE_LATENT_SIZE,                    hparams.moe_latent_size);

    // Sabaki stores a rotated latent K and a separate unrotated latent V.
    // Override the generic per-head dimensions used to size the unified KV cache.
    hparams.n_embd_head_k_full = hparams.n_lora_kv;
    hparams.n_embd_head_v_full = hparams.n_lora_kv;
    hparams.n_embd_head_k_swa  = hparams.n_lora_kv;
    hparams.n_embd_head_v_swa  = hparams.n_lora_kv;
    if (hparams.n_rot() != hparams.n_lora_kv || hparams.n_rot() % 2 != 0) {
        throw std::runtime_error("Sabaki RoPE must span the even-width MLA latent");
    }

    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_sabaki::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t latent   = hparams.n_lora_kv;
    const int64_t moe_dim  = hparams.moe_latent_size;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);

        if (hparams.is_recr(i)) {
            create_tensor_qkv(layer, i, n_embd, n_embd, n_embd, n_embd, 0);
            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {d_conv, 1, n_embd, 1}, 0);
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {d_conv, 1, n_embd, 1}, 0);
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {d_conv, 1, n_embd, 1}, 0);
            layer.ssm_f_a    = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", i), {n_embd, head_dim}, 0);
            layer.ssm_f_b    = create_tensor(tn(LLM_TENSOR_SSM_F_B, "weight", i), {head_dim, n_embd}, 0);
            layer.ssm_dt_b   = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {head_dim, n_head}, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, i), {n_head}, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {n_embd}, 0);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd}, 0);
            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, latent}, 0);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {latent}, 0);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {head_dim, latent, n_head}, 0);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {latent, head_dim, n_head}, 0);
        }
        layer.ssm_g_a = create_tensor(tn(LLM_TENSOR_SSM_G_A, "weight", i), {n_embd, n_embd}, 0);
        layer.wo      = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        if (i == 0) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
            layer.ffn_latent_down = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN, "weight", i), {n_embd, moe_dim}, 0);
            layer.ffn_norm_exps   = create_tensor(tn(LLM_TENSOR_FFN_NORM_EXPS, "weight", i), {moe_dim}, 0);
            layer.ffn_latent_up   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP, "weight", i), {moe_dim, n_embd}, 0);
            layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {moe_dim, hparams.n_ff_exp, n_expert}, 0);
            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {moe_dim, hparams.n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {hparams.n_ff_exp, moe_dim, n_expert}, 0);
            layer.ffn_gate_shexp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
            layer.ffn_up_shexp    = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
            layer.ffn_down_shexp  = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
        }
    }
}

bool llama_model_sabaki::load_tensors(llama_model_loader & ml) {
    if (!llama_model_base::load_tensors(ml)) return false;
    if (ml.no_alloc) return true;
    decay_exp.resize(hparams.n_layer(), nullptr);
    conv_qkv.resize(hparams.n_layer(), nullptr);
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const auto & layer = layers[il];
        if (!hparams.is_recr(il)) continue;
        ggml_init_params init = { 2 * ggml_tensor_overhead(), nullptr, true };
        ggml_context_ptr constants(ggml_init(init));
        if (!constants) throw std::runtime_error("Sabaki constant context allocation failed");
        auto * decay = ggml_new_tensor_1d(constants.get(), GGML_TYPE_F32, hparams.n_head());
        auto * conv = ggml_new_tensor_2d(constants.get(), GGML_TYPE_F32, hparams.ssm_d_conv, 3 * hparams.n_embd);
        ggml_format_name(decay, "blk.%u.ssm_decay_exp", il);
        ggml_format_name(conv, "blk.%u.ssm_conv_qkv", il);
        auto buft = ggml_backend_buffer_get_type(layer.ssm_a->buffer);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(constants.get(), buft));
        if (!buffer) throw std::runtime_error("Sabaki constant buffer allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> values(hparams.n_head());
        ggml_backend_tensor_get(layer.ssm_a, values.data(), 0, values.size() * sizeof(float));
        // Match the original CPU EXP exactly; GPU expf can change quantized routing.
        for (auto & value : values) value = expf(value);
        ggml_backend_tensor_set(decay, values.data(), 0, values.size() * sizeof(float));
        const size_t conv_size = ggml_nbytes(layer.ssm_q_conv);
        std::vector<uint8_t> packed(3 * conv_size);
        ggml_backend_tensor_get(layer.ssm_q_conv, packed.data(), 0, conv_size);
        ggml_backend_tensor_get(layer.ssm_k_conv, packed.data() + conv_size, 0, conv_size);
        ggml_backend_tensor_get(layer.ssm_v_conv, packed.data() + 2 * conv_size, 0, conv_size);
        ggml_backend_tensor_set(conv, packed.data(), 0, packed.size());
        decay_exp[il] = decay;
        conv_qkv[il] = conv;
        constant_contexts.emplace_back(std::move(constants));
        constant_buffers.emplace_back(std::move(buffer));
    }
    return true;
}

std::unique_ptr<llm_graph_context> llama_model_sabaki::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_sabaki::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_build_delta_net_base(params), model(model) {
    const auto & sabaki = static_cast<const llama_model_sabaki &>(model);
    const int64_t n_head = hparams.n_head();
    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_inner = n_head * head_dim;
    const int64_t d_conv = hparams.ssm_d_conv;
    const int64_t latent = hparams.n_lora_kv;
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // The tied output matrix is already on the output device.
    ggml_tensor * hidden = build_inp_embd(model.output);
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_kv = build_inp_mem_hybrid();
    auto * inp_rs = inp_kv->get_recr();
    auto * inp_attn_kv = inp_kv->get_attn();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * x = build_norm(hidden, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * cur = nullptr;

        if (hparams.is_recr(il)) {
            const auto * mctx = inp_rs->mctx;
            ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq, x);
            ggml_tensor * k = ggml_mul_mat(ctx0, layer.wk, x);
            ggml_tensor * v = ggml_mul_mat(ctx0, layer.wv, x);
            q = ggml_reshape_3d(ctx0, q, d_inner, n_seq_tokens, n_seqs);
            k = ggml_reshape_3d(ctx0, k, d_inner, n_seq_tokens, n_seqs);
            v = ggml_reshape_3d(ctx0, v, d_inner, n_seq_tokens, n_seqs);
            ggml_tensor * qkv = ggml_concat(ctx0, ggml_concat(ctx0, q, k, 0), v, 0);
            ggml_tensor * conv_input = build_conv_state(inp_rs, mctx->get_r_l(il), qkv, d_conv, 3*d_inner, il);
            ggml_tensor * qw = ggml_reshape_2d(ctx0, layer.ssm_q_conv, d_conv, d_inner);
            ggml_tensor * kw = ggml_reshape_2d(ctx0, layer.ssm_k_conv, d_conv, d_inner);
            ggml_tensor * vw = ggml_reshape_2d(ctx0, layer.ssm_v_conv, d_conv, d_inner);
            ggml_tensor * conv_w = sabaki.conv_qkv.empty() ? ggml_concat(ctx0, ggml_concat(ctx0, qw, kw, 1), vw, 1) : sabaki.conv_qkv[il];
            ggml_tensor * mixed = ggml_silu(ctx0, ggml_reshape_2d(ctx0, ggml_ssm_conv(ctx0, conv_input, conv_w), 3*d_inner, n_tokens));
            q = ggml_view_2d(ctx0, mixed, d_inner, n_tokens, mixed->nb[1], 0);
            k = ggml_view_2d(ctx0, mixed, d_inner, n_tokens, mixed->nb[1], ggml_row_size(mixed->type, d_inner));
            v = ggml_view_2d(ctx0, mixed, d_inner, n_tokens, mixed->nb[1], ggml_row_size(mixed->type, 2*d_inner));
            q = ggml_scale(ctx0, ggml_l2_norm(ctx0, ggml_cont_4d(ctx0, q, head_dim, n_head, n_seq_tokens, n_seqs), hparams.f_norm_rms_eps), std::sqrt((float) head_dim));
            k = ggml_l2_norm(ctx0, ggml_cont_4d(ctx0, k, head_dim, n_head, n_seq_tokens, n_seqs), hparams.f_norm_rms_eps);
            v = ggml_cont_4d(ctx0, v, head_dim, n_head, n_seq_tokens, n_seqs);

            ggml_tensor * z = ggml_mul_mat(ctx0, layer.ssm_f_b, ggml_mul_mat(ctx0, layer.ssm_f_a, x));
            z = ggml_add(ctx0, ggml_reshape_3d(ctx0, z, head_dim, n_head, n_tokens), layer.ssm_dt_b);
            ggml_tensor * decay = sabaki.decay_exp.empty() ? ggml_exp(ctx0, layer.ssm_a) : sabaki.decay_exp[il];
            ggml_tensor * decay_scale = ggml_reshape_3d(ctx0, decay, 1, n_head, 1);
            ggml_tensor * g = ggml_scale(ctx0, ggml_sigmoid(ctx0, ggml_mul(ctx0, z, decay_scale)), -5.0f);
            g = ggml_reshape_4d(ctx0, g, head_dim, n_head, n_seq_tokens, n_seqs);
            ggml_tensor * beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, layer.ssm_beta, x), 1, n_head, n_seq_tokens, n_seqs));
            ggml_tensor * state = build_rs(inp_rs, mctx->get_s_l(il), hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);
            cur = build_recurrent_attn(inp_rs, mctx->get_s_l(il), q, k, v, g, beta, state, il);
            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, cur), d_inner, n_tokens);
            cur = build_norm(cur, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
        } else {
            ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, layer.wq, x), head_dim, n_head, n_tokens);
            q = ggml_permute(ctx0, q, 0, 2, 1, 3);
            q = ggml_mul_mat(ctx0, layer.wk_b, q);
            q = ggml_permute(ctx0, q, 0, 2, 1, 3);
            ggml_tensor * kv = build_norm(ggml_mul_mat(ctx0, layer.wkv_a_mqa, x), layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            ggml_tensor * k = ggml_reshape_3d(ctx0, kv, latent, 1, n_tokens);
            ggml_tensor * v = k;
            q = ggml_rope_ext(ctx0, ggml_cont(ctx0, q), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            k = ggml_rope_ext(ctx0, k, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cur = build_attn(inp_attn_kv, nullptr, nullptr, nullptr, q, k, v, nullptr, nullptr, nullptr,
                    1.0f/std::sqrt((float) head_dim), il);
            // Decompress each head after attention over the unrotated V cache.
            cur = ggml_reshape_3d(ctx0, cur, latent, n_head, n_tokens);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, layer.wv_b, cur);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont_2d(ctx0, cur, d_inner, n_tokens);
        }
        ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, layer.ssm_g_a, x));
        cur = ggml_mul_mat(ctx0, layer.wo, ggml_mul(ctx0, cur, gate));
        hidden = ggml_add(ctx0, hidden, cur);

        x = build_norm(hidden, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        if (il == 0) {
            cur = build_ffn(x, layer.ffn_up, nullptr, nullptr, layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr, nullptr, LLM_FFN_SITU, LLM_FFN_PAR, il);
        } else {
            ggml_tensor * router = build_lora_mm(layer.ffn_gate_inp, x);
            ggml_tensor * lx = ggml_mul_mat(ctx0, layer.ffn_latent_down, x);
            ggml_tensor * routed = build_moe_ffn(lx, layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps,
                    layer.ffn_down_exps, layer.ffn_exp_probs_b, hparams.n_expert, hparams.n_expert_used,
                    LLM_FFN_SITU, true, 1.0f, LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il, router);
            routed = build_norm(routed, layer.ffn_norm_exps, nullptr, LLM_NORM_RMS, il);
            routed = ggml_mul_mat(ctx0, layer.ffn_latent_up, routed);
            ggml_tensor * shared = build_ffn(x, layer.ffn_up_shexp, nullptr, nullptr, layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr, nullptr, LLM_FFN_SITU, LLM_FFN_PAR, il);
            cur = ggml_add(ctx0, routed, shared);
        }
        hidden = ggml_add(ctx0, hidden, cur);
        cb(hidden, "l_out", il);
    }

    ggml_tensor * cur = build_norm(hidden, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    if (inp_out_ids) cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    res->t_embd = cur;
    cur = ggml_mul_mat(ctx0, model.output, cur);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
    // Small sigmoid-router margins can amplify reduced-precision matmul
    // accumulation into different expert choices. Preserve F32 arithmetic
    // for this model's GEMM/GEMV nodes; exported weights can remain F16.
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_MUL_MAT) {
            ggml_mul_mat_set_prec(node, GGML_PREC_F32);
        }
        // F16 matvec fusion reads GLU parameters as matmul precision. Keep these separate.
        if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[0]->type == GGML_TYPE_F16) {
            ggml_set_output(node);
        }
    }
}
