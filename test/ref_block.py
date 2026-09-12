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

def generate_reference(pimi_path, output_bin):
    weights = load_pimi_weights(pimi_path)

    seq_len = 4
    hidden_dim = 768
    num_heads = 16
    head_dim = 48

    # Deterministic input x [4, 768]
    np.random.seed(42)
    x = np.random.randn(seq_len, hidden_dim).astype(np.float32) * 0.1

    # Layer 0 weights
    ln1_w = weights["layers.0.ln_1.weight"]
    ln1_b = weights["layers.0.ln_1.bias"]
    q_w = weights["layers.0.attn.q_proj.weight"]     # [768, 768]
    k_w = weights["layers.0.attn.k_proj.weight"]     # [768, 768]
    v_w = weights["layers.0.attn.v_proj.weight"]     # [768, 768]
    out_w = weights["layers.0.attn.out_proj.weight"] # [768, 768]
    out_b = weights["layers.0.attn.out_proj.bias"]   # [768]

    ln2_w = weights["layers.0.ln_2.weight"]
    ln2_b = weights["layers.0.ln_2.bias"]
    fc_w = weights["layers.0.mlp.fc.weight"]         # [768, 3072]
    fc_b = weights["layers.0.mlp.fc.bias"]           # [3072]
    proj_w = weights["layers.0.mlp.proj.weight"]     # [3072, 768]
    proj_b = weights["layers.0.mlp.proj.bias"]       # [768]

    # Forward computation
    # 1. LN1
    ln1 = layernorm(x, ln1_w, ln1_b)

    # 2. Q, K, V projections
    Q = ln1 @ q_w
    K = ln1 @ k_w
    V = ln1 @ v_w

    # 3. Multi-head split
    # [seq, heads, head_dim] -> [heads, seq, head_dim]
    Q_heads = Q.reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)
    K_heads = K.reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)
    V_heads = V.reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)

    # 4. Scores: [heads, seq, seq]
    scores = (Q_heads @ K_heads.swapaxes(-1, -2)) / np.sqrt(head_dim)

    # 5. Causal mask
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            scores[:, i, j] = -1e9

    # 6. Softmax
    attn_weights = softmax(scores, axis=-1)

    # 7. Context: [heads, seq, head_dim] -> [seq, hidden]
    context = attn_weights @ V_heads
    context = context.swapaxes(0, 1).reshape(seq_len, hidden_dim)

    # 8. Output projection + bias
    attn_out = context @ out_w + out_b

    # 9. Residual 1
    res1 = x + attn_out

    # 10. LN2
    ln2 = layernorm(res1, ln2_w, ln2_b)

    # 11. MLP FC + bias + GELU
    mlp_fc = ln2 @ fc_w + fc_b
    mlp_gelu = gelu_new(mlp_fc)

    # 12. MLP Proj + bias
    mlp_proj = mlp_gelu @ proj_w + proj_b

    # 13. Residual 2
    out = res1 + mlp_proj

    # Save to binary file
    intermediates = [
        ("x", x),
        ("ln1", ln1),
        ("q", Q),
        ("k", K),
        ("v", V),
        ("attn_scores", scores),
        ("attn_weights", attn_weights),
        ("attn_out", attn_out),
        ("res1", res1),
        ("ln2", ln2),
        ("mlp_fc", mlp_fc),
        ("mlp_gelu", mlp_gelu),
        ("mlp_proj", mlp_proj),
        ("out", out),
    ]

    with open(output_bin, "wb") as f:
        f.write(struct.pack("<I", len(intermediates)))
        for name, arr in intermediates:
            name_b = name.encode("utf-8")[:31]
            name_buf = name_b + b"\x00" * (32 - len(name_b))
            shape = list(arr.shape) + [0] * (4 - len(arr.shape))
            f.write(name_buf)
            f.write(struct.pack("<4iQ", shape[0], shape[1], shape[2], shape[3], arr.size * 4))
            f.write(arr.astype(np.float32).tobytes())

    print(f"saved reference intermediates: {output_bin} ({len(intermediates)} tensors)")

if __name__ == "__main__":
    generate_reference("models/tinystories_33m.pimi", "test/ref_block_seq4.bin")
