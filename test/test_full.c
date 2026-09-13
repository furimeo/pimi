#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "pimi.h"

#pragma pack(push, 1)
typedef struct {
    char magic[4];
    unsigned int version;
    unsigned int num_tensors;
    unsigned long long total_weights_bytes;
    unsigned int vocab_size;
    unsigned int hidden_dim;
    unsigned int num_layers;
    unsigned int num_heads;
    unsigned int max_seq_len;
    char padding[24];
} PimiHeader;

typedef struct {
    char name[64];
    int ndim;
    int dims[4];
    unsigned long long offset;
    unsigned long long numel;
    unsigned long long bytes;
    unsigned int reserved[5];
} PimiTensorEntry;
#pragma pack(pop)

typedef struct {
    Tensor *ln1_w, *ln1_b;
    Tensor *q_w, *k_w, *v_w;
    Tensor *out_w, *out_b;
    Tensor *ln2_w, *ln2_b;
    Tensor *fc_w, *fc_b;
    Tensor *proj_w, *proj_b;
} LayerWeights;

static Tensor* load_tensor(FILE *f, PimiTensorEntry *entries, int num_tensors, const char *name, PimiDevice dev) {
    for (int i = 0; i < num_tensors; ++i) {
        if (strcmp(entries[i].name, name) == 0) {
            Tensor *t = pimi_tensor_new(entries[i].dims, entries[i].ndim, dev);
            if (!t) return NULL;

            fseek(f, (long)entries[i].offset, SEEK_SET);
            float *h_buf = (float*)malloc(entries[i].bytes);
            fread(h_buf, 1, entries[i].bytes, f);

            if (dev == PIMI_DEVICE_CPU) {
                memcpy(t->data, h_buf, entries[i].bytes);
            } else {
                Tensor *t_wrap = pimi_tensor_wrap(h_buf, entries[i].dims, entries[i].ndim, PIMI_DEVICE_CPU);
                pimi_tensor_copy(t, t_wrap);
                pimi_tensor_free(t_wrap);
            }
            free(h_buf);
            return t;
        }
    }
    fprintf(stderr, "error: missing weight: %s\n", name);
    return NULL;
}

int main(void) {
    printf("pimi full 4-layer forward test (seq=4, hidden=768)\n");

    const char *pimi_path = "models/tinystories_33m.pimi";
    FILE *f_pimi = fopen(pimi_path, "rb");
    if (!f_pimi) {
        pimi_path = "../models/tinystories_33m.pimi";
        f_pimi = fopen(pimi_path, "rb");
    }
    if (!f_pimi) {
        fprintf(stderr, "error: missing tinystories_33m.pimi\n");
        return 1;
    }

    PimiHeader hdr;
    fread(&hdr, 1, sizeof(hdr), f_pimi);
    PimiTensorEntry *entries = (PimiTensorEntry*)malloc(hdr.num_tensors * sizeof(PimiTensorEntry));
    fread(entries, sizeof(PimiTensorEntry), hdr.num_tensors, f_pimi);

    if (pimi_init("src/kernels.ptx") != 0) {
        fprintf(stderr, "error: pimi_init failed\n");
        return 1;
    }

    // Load embeddings on host CPU (for fast indexed lookup)
    Tensor *embed_w = load_tensor(f_pimi, entries, hdr.num_tensors, "embed.weight", PIMI_DEVICE_CPU);
    Tensor *pos_w   = load_tensor(f_pimi, entries, hdr.num_tensors, "pos_embed.weight", PIMI_DEVICE_CPU);

    // Load 4 layers to GPU
    LayerWeights layers[4];
    for (int l = 0; l < 4; ++l) {
        char name[128];
        #define LOAD_L(field, suffix) do { \
            snprintf(name, sizeof(name), "layers.%d.%s", l, suffix); \
            layers[l].field = load_tensor(f_pimi, entries, hdr.num_tensors, name, PIMI_DEVICE_CUDA); \
        } while (0)

        LOAD_L(ln1_w, "ln_1.weight");
        LOAD_L(ln1_b, "ln_1.bias");
        LOAD_L(q_w, "attn.q_proj.weight");
        LOAD_L(k_w, "attn.k_proj.weight");
        LOAD_L(v_w, "attn.v_proj.weight");
        LOAD_L(out_w, "attn.out_proj.weight");
        LOAD_L(out_b, "attn.out_proj.bias");

        LOAD_L(ln2_w, "ln_2.weight");
        LOAD_L(ln2_b, "ln_2.bias");
        LOAD_L(fc_w, "mlp.fc.weight");
        LOAD_L(fc_b, "mlp.fc.bias");
        LOAD_L(proj_w, "mlp.proj.weight");
        LOAD_L(proj_b, "mlp.proj.bias");
    }

    Tensor *ln_f_w = load_tensor(f_pimi, entries, hdr.num_tensors, "ln_f.weight", PIMI_DEVICE_CUDA);
    Tensor *ln_f_b = load_tensor(f_pimi, entries, hdr.num_tensors, "ln_f.bias", PIMI_DEVICE_CUDA);
    Tensor *lm_head_w = load_tensor(f_pimi, entries, hdr.num_tensors, "lm_head.weight", PIMI_DEVICE_CUDA);

    fclose(f_pimi);
    free(entries);

    // Read reference file
    const char *ref_path = "test/ref_full_seq4.bin";
    FILE *f_ref = fopen(ref_path, "rb");
    if (!f_ref) {
        ref_path = "ref_full_seq4.bin";
        f_ref = fopen(ref_path, "rb");
    }
    if (!f_ref) {
        fprintf(stderr, "error: missing ref_full_seq4.bin\n");
        return 1;
    }

    unsigned int ref_seq = 0, ref_vocab = 0;
    fread(&ref_seq, sizeof(unsigned int), 1, f_ref);
    fread(&ref_vocab, sizeof(unsigned int), 1, f_ref);

    int *token_ids = (int*)malloc(ref_seq * sizeof(int));
    fread(token_ids, sizeof(int), ref_seq, f_ref);

    float *ref_logits = (float*)malloc(ref_seq * ref_vocab * sizeof(float));
    fread(ref_logits, sizeof(float), ref_seq * ref_vocab, f_ref);
    fclose(f_ref);

    int seq_len = (int)ref_seq;
    int hidden_dim = 768;

    // 1. Host Embedding lookup: x = tok_emb + pos_emb
    int x_dims[2] = {seq_len, hidden_dim};
    int mlp_dims[2] = {seq_len, 3072};
    int logits_dims[2] = {seq_len, 50257};

    float *h_x_init = (float*)malloc(seq_len * hidden_dim * sizeof(float));
    const float *emb_raw = (const float*)embed_w->data;
    const float *pos_raw = (const float*)pos_w->data;

    for (int i = 0; i < seq_len; ++i) {
        int tid = token_ids[i];
        const float *tok_vec = &emb_raw[tid * hidden_dim];
        const float *pos_vec = &pos_raw[i * hidden_dim];
        float *x_vec = &h_x_init[i * hidden_dim];
        for (int d = 0; d < hidden_dim; ++d) {
            x_vec[d] = tok_vec[d] + pos_vec[d];
        }
    }

    // Allocate GPU activation buffers (pre-allocated static buffers)
    Tensor *x = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *ln1 = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *q = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *k = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *v = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *context = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *attn_out = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *res1 = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *ln2 = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *mlp_fc = pimi_tensor_new(mlp_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *mlp_gelu = pimi_tensor_new(mlp_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *mlp_proj = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *logits = pimi_tensor_new(logits_dims, 2, PIMI_DEVICE_CUDA);

    // Copy initial x [4, 768] to GPU
    Tensor *h_wrap_x = pimi_tensor_wrap(h_x_init, x_dims, 2, PIMI_DEVICE_CPU);
    pimi_tensor_copy(x, h_wrap_x);
    pimi_tensor_free(h_wrap_x);
    free(h_x_init);

    printf("running 4 transformer blocks on gpu...\n");

    // Forward through all 4 layers entirely on GPU
    for (int l = 0; l < 4; ++l) {
        pimi_layernorm(ln1, x, layers[l].ln1_w, layers[l].ln1_b, 1e-5f);
        pimi_matmul(q, ln1, layers[l].q_w);
        pimi_matmul(k, ln1, layers[l].k_w);
        pimi_matmul(v, ln1, layers[l].v_w);
        pimi_attn_causal(context, q, k, v, 16);
        pimi_matmul(attn_out, context, layers[l].out_w);
        pimi_add_bias(attn_out, layers[l].out_b);
        pimi_add(res1, x, attn_out);

        pimi_layernorm(ln2, res1, layers[l].ln2_w, layers[l].ln2_b, 1e-5f);
        pimi_matmul(mlp_fc, ln2, layers[l].fc_w);
        pimi_add_bias(mlp_fc, layers[l].fc_b);
        pimi_gelu(mlp_gelu, mlp_fc);
        pimi_matmul(mlp_proj, mlp_gelu, layers[l].proj_w);
        pimi_add_bias(mlp_proj, layers[l].proj_b);
        pimi_add(x, res1, mlp_proj); // update x for next layer
    }

    // Final LayerNorm
    pimi_layernorm(x, x, ln_f_w, ln_f_b, 1e-5f);

    // LM Head
    pimi_matmul(logits, x, lm_head_w);

    // Copy logits back to CPU
    Tensor *h_logits = pimi_tensor_new(logits_dims, 2, PIMI_DEVICE_CPU);
    pimi_tensor_copy(h_logits, logits);

    // Compare with reference
    float max_diff = 0.0f;
    const float *pimi_out = (const float*)h_logits->data;
    size_t total_logits = (size_t)seq_len * 50257;
    for (size_t i = 0; i < total_logits; ++i) {
        float d = fabsf(pimi_out[i] - ref_logits[i]);
        if (d > max_diff) max_diff = d;
    }

    printf("\n4-layer full forward result:\n");
    printf("  max_abs_diff: %.6f (%s)\n", max_diff, (max_diff < 1e-3f) ? "ok" : "fail");

    // Verify top-5 tokens for last position
    int last_pos_offset = (seq_len - 1) * 50257;
    Tensor *last_logit_wrap = pimi_tensor_wrap((void*)&pimi_out[last_pos_offset], &hdr.vocab_size, 1, PIMI_DEVICE_CPU);
    Tensor *probs = pimi_tensor_new(&hdr.vocab_size, 1, PIMI_DEVICE_CPU);
    pimi_softmax_cpu(probs, last_logit_wrap);

    float *prob_data = (float*)probs->data;
    printf("\ntop 5 predicted tokens on quadro 2000:\n");
    for (int k = 0; k < 5; ++k) {
        int best = 0;
        float best_p = -1.0f;
        for (unsigned int i = 0; i < hdr.vocab_size; ++i) {
            if (prob_data[i] > best_p) {
                best_p = prob_data[i];
                best = (int)i;
            }
        }
        printf("  [%d] token %5d: prob=%.4f (logit=%.3f)\n",
               k + 1, best, best_p, pimi_out[last_pos_offset + best]);
        prob_data[best] = -1.0f; // mask out to find next best
    }

    // Cleanup
    pimi_tensor_free(embed_w); pimi_tensor_free(pos_w);
    for (int l = 0; l < 4; ++l) {
        pimi_tensor_free(layers[l].ln1_w); pimi_tensor_free(layers[l].ln1_b);
        pimi_tensor_free(layers[l].q_w); pimi_tensor_free(layers[l].k_w); pimi_tensor_free(layers[l].v_w);
        pimi_tensor_free(layers[l].out_w); pimi_tensor_free(layers[l].out_b);
        pimi_tensor_free(layers[l].ln2_w); pimi_tensor_free(layers[l].ln2_b);
        pimi_tensor_free(layers[l].fc_w); pimi_tensor_free(layers[l].fc_b);
        pimi_tensor_free(layers[l].proj_w); pimi_tensor_free(layers[l].proj_b);
    }
    pimi_tensor_free(ln_f_w); pimi_tensor_free(ln_f_b); pimi_tensor_free(lm_head_w);

    pimi_tensor_free(x); pimi_tensor_free(ln1); pimi_tensor_free(q); pimi_tensor_free(k); pimi_tensor_free(v);
    pimi_tensor_free(context); pimi_tensor_free(attn_out); pimi_tensor_free(res1);
    pimi_tensor_free(ln2); pimi_tensor_free(mlp_fc); pimi_tensor_free(mlp_gelu); pimi_tensor_free(mlp_proj);
    pimi_tensor_free(logits); pimi_tensor_free(h_logits); pimi_tensor_free(last_logit_wrap); pimi_tensor_free(probs);

    free(token_ids); free(ref_logits);
    pimi_shutdown();
    return 0;
}
