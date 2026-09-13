#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

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

struct GptNeoModel {
    GptNeoConfig cfg;
    int max_seq;

    // Weights
    Tensor *embed_w;
    Tensor *pos_w;
    LayerWeights layers[4];
    Tensor *ln_f_w, *ln_f_b;
    Tensor *lm_head_w;

    // Pre-allocated static activation buffers (CUDA)
    Tensor *buf_x;
    Tensor *buf_ln1;
    Tensor *buf_q, *buf_k, *buf_v;
    Tensor *buf_context;
    Tensor *buf_attn_out;
    Tensor *buf_res1;
    Tensor *buf_ln2;
    Tensor *buf_mlp_fc;
    Tensor *buf_mlp_gelu;
    Tensor *buf_mlp_proj;

    // Last-token projection buffers
    Tensor *buf_last_x;      // [1, H] on CUDA (wrapped pointer)
    Tensor *buf_last_logits; // [1, V] on CUDA
    Tensor *h_last_logits;   // [1, V] on CPU
    Tensor *h_staging_emb;   // [max_seq, H] on CPU
};

static Tensor* load_tensor(FILE *f, PimiTensorEntry *entries, int num_tensors, const char *name, PimiDevice dev) {
    for (int i = 0; i < num_tensors; ++i) {
        if (strcmp(entries[i].name, name) == 0) {
            Tensor *t = pimi_tensor_new(entries[i].dims, entries[i].ndim, dev);
            if (!t) return NULL;

            fseek(f, (long)entries[i].offset, SEEK_SET);
            float *h_buf = (float*)malloc(entries[i].bytes);
            if (!h_buf) {
                pimi_tensor_free(t);
                return NULL;
            }
            if (fread(h_buf, 1, entries[i].bytes, f) != entries[i].bytes) {
                free(h_buf);
                pimi_tensor_free(t);
                return NULL;
            }

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

GptNeoModel *model_load(const char *pimi_path, const char *ptx_path, int max_seq) {
    if (pimi_init(ptx_path) != 0) {
        fprintf(stderr, "error: failed to initialize pimi runtime (%s)\n", ptx_path);
        return NULL;
    }

    FILE *f = fopen(pimi_path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", pimi_path);
        return NULL;
    }

    PimiHeader hdr;
    if (fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr) || memcmp(hdr.magic, "PIMI", 4) != 0) {
        fprintf(stderr, "error: invalid pimi header\n");
        fclose(f);
        return NULL;
    }

    PimiTensorEntry *entries = (PimiTensorEntry*)malloc(hdr.num_tensors * sizeof(PimiTensorEntry));
    if (fread(entries, sizeof(PimiTensorEntry), hdr.num_tensors, f) != hdr.num_tensors) {
        fprintf(stderr, "error: failed to read tensor table\n");
        free(entries);
        fclose(f);
        return NULL;
    }

    GptNeoModel *m = (GptNeoModel*)calloc(1, sizeof(GptNeoModel));
    if (!m) {
        free(entries);
        fclose(f);
        return NULL;
    }

    m->cfg.vocab_size  = (int)hdr.vocab_size;
    m->cfg.hidden_dim  = (int)hdr.hidden_dim;
    m->cfg.num_layers  = (int)hdr.num_layers;
    m->cfg.num_heads   = (int)hdr.num_heads;
    m->cfg.max_seq_len = (int)hdr.max_seq_len;
    m->max_seq         = (max_seq > 0 && max_seq <= m->cfg.max_seq_len) ? max_seq : 256;

    int H = m->cfg.hidden_dim;
    int V = m->cfg.vocab_size;

    // Load embeddings on host CPU
    m->embed_w = load_tensor(f, entries, hdr.num_tensors, "embed.weight", PIMI_DEVICE_CPU);
    m->pos_w   = load_tensor(f, entries, hdr.num_tensors, "pos_embed.weight", PIMI_DEVICE_CPU);

    // Load 4 layers onto GPU
    for (int l = 0; l < m->cfg.num_layers; ++l) {
        char name[128];
        #define LOAD_L(field, suffix) do { \
            snprintf(name, sizeof(name), "layers.%d.%s", l, suffix); \
            m->layers[l].field = load_tensor(f, entries, hdr.num_tensors, name, PIMI_DEVICE_CUDA); \
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

    m->ln_f_w    = load_tensor(f, entries, hdr.num_tensors, "ln_f.weight", PIMI_DEVICE_CUDA);
    m->ln_f_b    = load_tensor(f, entries, hdr.num_tensors, "ln_f.bias", PIMI_DEVICE_CUDA);
    m->lm_head_w = load_tensor(f, entries, hdr.num_tensors, "lm_head.weight", PIMI_DEVICE_CUDA);

    fclose(f);
    free(entries);

    // Pre-allocate GPU activation buffers for max_seq
    int dims_h[2]   = {m->max_seq, H};
    int dims_mlp[2] = {m->max_seq, 3072};
    int dims_last_x[2] = {1, H};
    int dims_last_logits[2] = {1, V};

    m->buf_x        = pimi_tensor_new(dims_h, 2, PIMI_DEVICE_CUDA);
    m->buf_ln1      = pimi_tensor_new(dims_h, 2, PIMI_DEVICE_CUDA);
    m->buf_q        = pimi_tensor_new(dims_h, 2, PIMI_DEVICE_CUDA);
    m->buf_k        = pimi_tensor_new(dims_h, 2, PIMI_DEVICE_CUDA);
    m->buf_v        = pimi_tensor_new(dims_h, 2, PIMI_DEVICE_CUDA);
    m->buf_context  = pimi_tensor_new(dims_h, 2, PIMI_DEVICE_CUDA);
    m->buf_attn_out = pimi_tensor_new(dims_h, 2, PIMI_DEVICE_CUDA);
    m->buf_res1     = pimi_tensor_new(dims_h, 2, PIMI_DEVICE_CUDA);
    m->buf_ln2      = pimi_tensor_new(dims_h, 2, PIMI_DEVICE_CUDA);
    m->buf_mlp_fc   = pimi_tensor_new(dims_mlp, 2, PIMI_DEVICE_CUDA);
    m->buf_mlp_gelu = pimi_tensor_new(dims_mlp, 2, PIMI_DEVICE_CUDA);
    m->buf_mlp_proj = pimi_tensor_new(dims_h, 2, PIMI_DEVICE_CUDA);

    // Last token buffers
    m->buf_last_x      = pimi_tensor_new(dims_last_x, 2, PIMI_DEVICE_CUDA);
    m->buf_last_logits = pimi_tensor_new(dims_last_logits, 2, PIMI_DEVICE_CUDA);
    m->h_last_logits   = pimi_tensor_new(dims_last_logits, 2, PIMI_DEVICE_CPU);
    m->h_staging_emb   = pimi_tensor_new(dims_h, 2, PIMI_DEVICE_CPU);


    return m;
}

void model_free(GptNeoModel *m) {
    if (!m) return;

    pimi_tensor_free(m->embed_w);
    pimi_tensor_free(m->pos_w);

    for (int l = 0; l < m->cfg.num_layers; ++l) {
        pimi_tensor_free(m->layers[l].ln1_w);
        pimi_tensor_free(m->layers[l].ln1_b);
        pimi_tensor_free(m->layers[l].q_w);
        pimi_tensor_free(m->layers[l].k_w);
        pimi_tensor_free(m->layers[l].v_w);
        pimi_tensor_free(m->layers[l].out_w);
        pimi_tensor_free(m->layers[l].out_b);
        pimi_tensor_free(m->layers[l].ln2_w);
        pimi_tensor_free(m->layers[l].ln2_b);
        pimi_tensor_free(m->layers[l].fc_w);
        pimi_tensor_free(m->layers[l].fc_b);
        pimi_tensor_free(m->layers[l].proj_w);
        pimi_tensor_free(m->layers[l].proj_b);
    }

    pimi_tensor_free(m->ln_f_w);
    pimi_tensor_free(m->ln_f_b);
    pimi_tensor_free(m->lm_head_w);

    pimi_tensor_free(m->buf_x);
    pimi_tensor_free(m->buf_ln1);
    pimi_tensor_free(m->buf_q);
    pimi_tensor_free(m->buf_k);
    pimi_tensor_free(m->buf_v);
    pimi_tensor_free(m->buf_context);
    pimi_tensor_free(m->buf_attn_out);
    pimi_tensor_free(m->buf_res1);
    pimi_tensor_free(m->buf_ln2);
    pimi_tensor_free(m->buf_mlp_fc);
    pimi_tensor_free(m->buf_mlp_gelu);
    pimi_tensor_free(m->buf_mlp_proj);

    pimi_tensor_free(m->buf_last_x);
    pimi_tensor_free(m->buf_last_logits);
    pimi_tensor_free(m->h_last_logits);
    pimi_tensor_free(m->h_staging_emb);

    free(m);
    pimi_shutdown();
}

int model_vocab_size(const GptNeoModel *m) {
    return m ? m->cfg.vocab_size : 0;
}


float *model_forward(GptNeoModel *m, const int *tokens, int seq_len) {
    if (!m || !tokens || seq_len <= 0 || seq_len > m->max_seq) {
        fprintf(stderr, "error: invalid forward arguments (seq_len=%d, max=%d)\n", seq_len, m ? m->max_seq : 0);
        return NULL;
    }

    int H = m->cfg.hidden_dim;

    // 1. Embedding lookup on Host CPU
    float *h_emb = (float*)m->h_staging_emb->data;
    const float *tok_w = (const float*)m->embed_w->data;
    const float *pos_w = (const float*)m->pos_w->data;

    for (int i = 0; i < seq_len; ++i) {
        int tid = tokens[i];
        if (tid < 0 || tid >= m->cfg.vocab_size) tid = 0;
        const float *t_vec = &tok_w[tid * H];
        const float *p_vec = &pos_w[i * H];
        float *out_vec = &h_emb[i * H];
        for (int d = 0; d < H; ++d) {
            out_vec[d] = t_vec[d] + p_vec[d];
        }
    }

    // 2. Adjust dynamic dimensions
    m->buf_x->dims[0]        = seq_len; m->buf_x->numel        = (size_t)seq_len * H;
    m->buf_ln1->dims[0]      = seq_len; m->buf_ln1->numel      = (size_t)seq_len * H;
    m->buf_q->dims[0]        = seq_len; m->buf_q->numel        = (size_t)seq_len * H;
    m->buf_k->dims[0]        = seq_len; m->buf_k->numel        = (size_t)seq_len * H;
    m->buf_v->dims[0]        = seq_len; m->buf_v->numel        = (size_t)seq_len * H;
    m->buf_context->dims[0]  = seq_len; m->buf_context->numel  = (size_t)seq_len * H;
    m->buf_attn_out->dims[0] = seq_len; m->buf_attn_out->numel = (size_t)seq_len * H;
    m->buf_res1->dims[0]     = seq_len; m->buf_res1->numel     = (size_t)seq_len * H;
    m->buf_ln2->dims[0]      = seq_len; m->buf_ln2->numel      = (size_t)seq_len * H;
    m->buf_mlp_fc->dims[0]   = seq_len; m->buf_mlp_fc->numel   = (size_t)seq_len * 3072;
    m->buf_mlp_gelu->dims[0] = seq_len; m->buf_mlp_gelu->numel = (size_t)seq_len * 3072;
    m->buf_mlp_proj->dims[0] = seq_len; m->buf_mlp_proj->numel = (size_t)seq_len * H;

    // Copy embeddings to GPU
    m->h_staging_emb->dims[0] = seq_len;
    m->h_staging_emb->numel   = (size_t)seq_len * H;
    pimi_tensor_copy(m->buf_x, m->h_staging_emb);

    // 3. 4-Layer forward on GPU
    for (int l = 0; l < m->cfg.num_layers; ++l) {
        pimi_layernorm(m->buf_ln1, m->buf_x, m->layers[l].ln1_w, m->layers[l].ln1_b, 1e-5f);
        pimi_matmul(m->buf_q, m->buf_ln1, m->layers[l].q_w);
        pimi_matmul(m->buf_k, m->buf_ln1, m->layers[l].k_w);
        pimi_matmul(m->buf_v, m->buf_ln1, m->layers[l].v_w);
        pimi_attn_causal(m->buf_context, m->buf_q, m->buf_k, m->buf_v, m->cfg.num_heads);
        pimi_matmul(m->buf_attn_out, m->buf_context, m->layers[l].out_w);
        pimi_add_bias(m->buf_attn_out, m->layers[l].out_b);
        pimi_add(m->buf_res1, m->buf_x, m->buf_attn_out);

        pimi_layernorm(m->buf_ln2, m->buf_res1, m->layers[l].ln2_w, m->layers[l].ln2_b, 1e-5f);
        pimi_matmul(m->buf_mlp_fc, m->buf_ln2, m->layers[l].fc_w);
        pimi_add_bias(m->buf_mlp_fc, m->layers[l].fc_b);
        pimi_gelu(m->buf_mlp_gelu, m->buf_mlp_fc);
        pimi_matmul(m->buf_mlp_proj, m->buf_mlp_gelu, m->layers[l].proj_w);
        pimi_add_bias(m->buf_mlp_proj, m->layers[l].proj_b);
        pimi_add(m->buf_x, m->buf_res1, m->buf_mlp_proj);
    }

    // 4. Final LayerNorm
    pimi_layernorm(m->buf_x, m->buf_x, m->ln_f_w, m->ln_f_b, 1e-5f);

    // 5. Copy last token's hidden state to buf_last_x on GPU
    void *last_row_addr = (void*)((uintptr_t)m->buf_x->data + (size_t)(seq_len - 1) * H * sizeof(float));
    int dims_last[2] = {1, H};
    Tensor *t_slice = pimi_tensor_wrap(last_row_addr, dims_last, 2, PIMI_DEVICE_CUDA);
    pimi_tensor_copy(m->buf_last_x, t_slice);
    pimi_tensor_free(t_slice);

    // 6. Project [1, H] x [H, V] -> [1, V] on GPU
    pimi_matmul(m->buf_last_logits, m->buf_last_x, m->lm_head_w);

    // 7. Copy logits [1, V] back to CPU (200 KB)
    pimi_tensor_copy(m->h_last_logits, m->buf_last_logits);

    return (float*)m->h_last_logits->data;
}

