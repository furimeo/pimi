#!/usr/bin/env python3
import os
import sys
import numpy as np
from test_tok import encode, decode, vocab
from ref_full import load_pimi_weights, run_gpt_neo_block, layernorm

def run_block_no_scale(x, layer_idx, weights, seq_len, num_heads=16, head_dim=48):
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

    ln1 = layernorm(x, ln1_w, ln1_b)
    Q = ln1 @ q_w
    K = ln1 @ k_w
    V = ln1 @ v_w

    Q_heads = Q.reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)
    K_heads = K.reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)
    V_heads = V.reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)

    # GPT-Neo: raw Q @ K^T WITHOUT sqrt(head_dim) scaling!
    scores = Q_heads @ K_heads.swapaxes(-1, -2)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            scores[:, i, j] = -1e9

    from ref_full import softmax, gelu_new
    attn_weights = softmax(scores, axis=-1)
    context = attn_weights @ V_heads
    context = context.swapaxes(0, 1).reshape(seq_len, num_heads * head_dim)

    attn_out = context @ out_w + out_b
    res1 = x + attn_out

    ln2 = layernorm(res1, ln2_w, ln2_b)
    mlp_fc = ln2 @ fc_w + fc_b
    mlp_gelu = gelu_new(mlp_fc)
    mlp_proj = mlp_gelu @ proj_w + proj_b

    return res1 + mlp_proj

def generate_numpy(pimi_path, prompt, max_tokens=40, repeat_penalty=1.0):
    weights = load_pimi_weights(pimi_path)
    embed_w = weights["embed.weight"]
    pos_w = weights["pos_embed.weight"]
    ln_f_w = weights["ln_f.weight"]
    ln_f_b = weights["ln_f.bias"]
    lm_head_w = weights["lm_head.weight"]

    tokens = encode(prompt)
    print("Initial tokens:", tokens, "->", repr(prompt))

    for step in range(max_tokens):
        seq_len = len(tokens)
        tok_emb = embed_w[tokens]
        pos_emb = pos_w[np.arange(seq_len)]
        x = tok_emb + pos_emb

        for l in range(4):
            x = run_block_no_scale(x, l, weights, seq_len)


        x = layernorm(x, ln_f_w, ln_f_b)
        last_x = x[-1:] # [1, 768]
        logits = (last_x @ lm_head_w)[0].copy() # [50257]

        # Apply repetition penalty
        if repeat_penalty != 1.0:
            for t in set(tokens):
                if logits[t] > 0:
                    logits[t] /= repeat_penalty
                else:
                    logits[t] *= repeat_penalty

        next_tok = int(np.argmax(logits))
        tokens.append(next_tok)
        print(f"Step {step+1}: next_tok={next_tok} ({repr(decode([next_tok]))})")

    print("\nFull generated text:\n" + decode(tokens))

if __name__ == "__main__":
    generate_numpy("models/tinystories_33m.pimi", "Once upon a time", 40, repeat_penalty=1.1)



