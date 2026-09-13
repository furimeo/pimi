#!/usr/bin/env python3
import os
import struct
import numpy as np

def gelu_new(x):
    s = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(s * (x + 0.044715 * np.power(x, 3))))

def layernorm(x, weight, bias, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def load_pimi_weights(pimi_path):
    with open(pimi_path, "rb") as f:
        hdr = f.read(64)
        magic, ver, num_t = struct.unpack("<4sII", hdr[:12])
        if magic != b"PIMI":
            raise ValueError("not a pimi file")

        entries = {}
        for _ in range(num_t):
            raw = f.read(128)
            name = raw[:64].split(b"\x00")[0].decode("utf-8")
            ndim, d0, d1, d2, d3 = struct.unpack("<i4i", raw[64:84])
            offset, numel, nbytes = struct.unpack("<QQQ", raw[84:108])
            dims = [d0, d1, d2, d3][:ndim]
            entries[name] = {"dims": dims, "offset": offset, "bytes": nbytes}

        weights = {}
        for name, meta in entries.items():
            f.seek(meta["offset"])
            data = f.read(meta["bytes"])
            arr = np.frombuffer(data, dtype=np.float32).reshape(meta["dims"])
            weights[name] = arr
        return weights

def run_gpt_neo_block(x, layer_idx, weights, seq_len, num_heads=16, head_dim=48):
    prefix = f"layers.{layer_idx}."
    ln1_w = weights[f"{prefix}ln_1.weight"]
    ln1_b = weights[f"{prefix}ln_1.bias"]
    q_w = weights[f"{prefix}attn.q_proj.weight"]
    k_w = weights[f"{prefix}attn.k_proj.weight"]
    v_w = weights[f"{prefix}attn.v_proj.weight"]
    out_w = weights[f"{prefix}attn.out_proj.weight"]
    out_b = weights[f"{prefix}attn.out_proj.bias"]

    ln2_w = weights[f"{prefix}ln_2.weight"]
    ln2_b = weights[f"{prefix}ln_2.bias"]
    fc_w = weights[f"{prefix}mlp.fc.weight"]
    fc_b = weights[f"{prefix}mlp.fc.bias"]
    proj_w = weights[f"{prefix}mlp.proj.weight"]
    proj_b = weights[f"{prefix}mlp.proj.bias"]

    # LN1
    ln1 = layernorm(x, ln1_w, ln1_b)

    # Q, K, V
    Q = ln1 @ q_w
    K = ln1 @ k_w
    V = ln1 @ v_w

    Q_heads = Q.reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)
    K_heads = K.reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)
    V_heads = V.reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)

    scores = (Q_heads @ K_heads.swapaxes(-1, -2)) / np.sqrt(head_dim)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            scores[:, i, j] = -1e9

    attn_weights = softmax(scores, axis=-1)
    context = attn_weights @ V_heads
    context = context.swapaxes(0, 1).reshape(seq_len, num_heads * head_dim)

    attn_out = context @ out_w + out_b
    res1 = x + attn_out

    # LN2
    ln2 = layernorm(res1, ln2_w, ln2_b)

    # MLP
    mlp_fc = ln2 @ fc_w + fc_b
    mlp_gelu = gelu_new(mlp_fc)
    mlp_proj = mlp_gelu @ proj_w + proj_b

    out = res1 + mlp_proj
    return out

def full_forward(pimi_path, token_ids, output_bin):
    weights = load_pimi_weights(pimi_path)
    seq_len = len(token_ids)

    # 1. Embedding lookup
    embed_w = weights["embed.weight"]         # [50257, 768]
    pos_w = weights["pos_embed.weight"]       # [2048, 768]
    tok_emb = embed_w[token_ids]              # [seq, 768]
    pos_emb = pos_w[np.arange(seq_len)]       # [seq, 768]
    x = tok_emb + pos_emb                     # [seq, 768]

    # 2. Blocks 0..3
    for layer in range(4):
        x = run_gpt_neo_block(x, layer, weights, seq_len)

    # 3. Final LN
    ln_f_w = weights["ln_f.weight"]
    ln_f_b = weights["ln_f.bias"]
    x = layernorm(x, ln_f_w, ln_f_b)

    # 4. LM Head projection
    lm_head_w = weights["lm_head.weight"]     # [768, 50257]
    logits = x @ lm_head_w                    # [seq, 50257]

    # Save to binary
    with open(output_bin, "wb") as f:
        f.write(struct.pack("<II", seq_len, 50257))
        f.write(np.array(token_ids, dtype=np.int32).tobytes())
        f.write(logits.astype(np.float32).tobytes())

    # Top-5 tokens for last position
    last_logits = logits[-1]
    probs = softmax(last_logits)
    top5_idx = np.argsort(probs)[-5:][::-1]

    print(f"reference full forward completed for {seq_len} tokens")
    print(f"saved: {output_bin}")
    print("top 5 predicted next tokens:")
    for idx in top5_idx:
        print(f"  token {idx:>5}: prob={probs[idx]:.4f} (logit={last_logits[idx]:.3f})")

if __name__ == "__main__":
    # Test with 4 token IDs (e.g. "Once upon a time" -> typical start)
    test_tokens = [1507, 2402, 257, 640] # Once upon a time
    full_forward("models/tinystories_33m.pimi", test_tokens, "test/ref_full_seq4.bin")
