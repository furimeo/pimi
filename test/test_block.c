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

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static Tensor* load_weight_to_gpu(FILE *f, PimiTensorEntry *entries, int num_tensors, const char *target_name) {
    for (int i = 0; i < num_tensors; ++i) {
        if (strcmp(entries[i].name, target_name) == 0) {
            Tensor *t_gpu = pimi_tensor_new(entries[i].dims, entries[i].ndim, PIMI_DEVICE_CUDA);
            if (!t_gpu) return NULL;

            fseek(f, (long)entries[i].offset, SEEK_SET);
            float *h_buf = (float*)malloc(entries[i].bytes);
            fread(h_buf, 1, entries[i].bytes, f);

            Tensor *t_cpu = pimi_tensor_wrap(h_buf, entries[i].dims, entries[i].ndim, PIMI_DEVICE_CPU);
            pimi_tensor_copy(t_gpu, t_cpu);

            pimi_tensor_free(t_cpu);
            free(h_buf);
            return t_gpu;
        }
    }
    fprintf(stderr, "error: weight not found: %s\n", target_name);
    return NULL;
}

typedef struct {
    char name[32];
    int dims[4];
    size_t bytes;
    float *data;
} RefTensor;

static RefTensor* load_reference_file(const char *path, int *count_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    unsigned int count = 0;
    fread(&count, sizeof(unsigned int), 1, f);
    RefTensor *refs = (RefTensor*)malloc(count * sizeof(RefTensor));

    for (unsigned int i = 0; i < count; ++i) {
        fread(refs[i].name, 1, 32, f);
        unsigned long long nbytes = 0;
        fread(refs[i].dims, sizeof(int), 4, f);
        fread(&nbytes, sizeof(unsigned long long), 1, f);
        refs[i].bytes = (size_t)nbytes;
        refs[i].data = (float*)malloc(refs[i].bytes);
        fread(refs[i].data, 1, refs[i].bytes, f);
    }
    fclose(f);
    *count_out = (int)count;
    return refs;
}

static const float* find_ref(RefTensor *refs, int count, const char *name) {
    for (int i = 0; i < count; ++i) {
        if (strcmp(refs[i].name, name) == 0) return refs[i].data;
    }
    return NULL;
}

static void free_refs(RefTensor *refs, int count) {
    for (int i = 0; i < count; ++i) free(refs[i].data);
    free(refs);
}

int main(void) {
    printf("pimi gpt-neo block 0 unit test (seq=4, hidden=768)\n");

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

    int ref_count = 0;
    const char *ref_path = "test/ref_block_seq4.bin";
    RefTensor *refs = load_reference_file(ref_path, &ref_count);
    if (!refs) {
        ref_path = "ref_block_seq4.bin";
        refs = load_reference_file(ref_path, &ref_count);
    }
    if (!refs) {
        fprintf(stderr, "error: missing ref_block_seq4.bin\n");
        return 1;
    }

    // Load Layer 0 weights to GPU
    Tensor *ln1_w = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.ln_1.weight");
    Tensor *ln1_b = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.ln_1.bias");
    Tensor *q_w   = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.attn.q_proj.weight");
    Tensor *k_w   = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.attn.k_proj.weight");
    Tensor *v_w   = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.attn.v_proj.weight");
    Tensor *out_w = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.attn.out_proj.weight");
    Tensor *out_b = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.attn.out_proj.bias");

    Tensor *ln2_w  = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.ln_2.weight");
    Tensor *ln2_b  = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.ln_2.bias");
    Tensor *fc_w   = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.mlp.fc.weight");
    Tensor *fc_b   = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.mlp.fc.bias");
    Tensor *proj_w = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.mlp.proj.weight");
    Tensor *proj_b = load_weight_to_gpu(f_pimi, entries, hdr.num_tensors, "layers.0.mlp.proj.bias");

    fclose(f_pimi);
    free(entries);

    // Setup input x [4, 768] on GPU
    int x_dims[2] = {4, 768};
    int mlp_dims[2] = {4, 3072};
    Tensor *x = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);

    const float *ref_x = find_ref(refs, ref_count, "x");
    Tensor *h_x = pimi_tensor_wrap((void*)ref_x, x_dims, 2, PIMI_DEVICE_CPU);
    pimi_tensor_copy(x, h_x);
    pimi_tensor_free(h_x);

    // Intermediate tensors on GPU
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
    Tensor *out = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);

    // Host staging buffer for verification
    Tensor *h_check = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CPU);
    Tensor *h_check_mlp = pimi_tensor_new(mlp_dims, 2, PIMI_DEVICE_CPU);

    printf("\n%-20s | %-12s | %s\n", "stage", "max_abs_diff", "status");
    printf("---------------------+--------------+-------\n");

    #define CHECK(stage_name, gpu_tensor, host_staging, ref_name, num_elements) do { \
        pimi_tensor_copy(host_staging, gpu_tensor); \
        const float *ref_ptr = find_ref(refs, ref_count, ref_name); \
        float diff = max_abs_diff((const float*)host_staging->data, ref_ptr, num_elements); \
        printf("%-20s | %12.6f | %s\n", stage_name, diff, (diff < 1e-3f) ? "ok" : "fail"); \
    } while (0)

    // 1. LayerNorm 1
    pimi_layernorm(ln1, x, ln1_w, ln1_b, 1e-5f);
    CHECK("1. ln1", ln1, h_check, "ln1", 4 * 768);

    // 2. Q, K, V
    pimi_matmul(q, ln1, q_w);
    CHECK("2. q_proj", q, h_check, "q", 4 * 768);

    pimi_matmul(k, ln1, k_w);
    CHECK("3. k_proj", k, h_check, "k", 4 * 768);

    pimi_matmul(v, ln1, v_w);
    CHECK("4. v_proj", v, h_check, "v", 4 * 768);

    // 3. Multi-Head Causal Attention
    pimi_attn_causal(context, q, k, v, 16);

    // 4. Output projection + bias
    pimi_matmul(attn_out, context, out_w);
    pimi_add_bias(attn_out, out_b);
    CHECK("5. attn_out", attn_out, h_check, "attn_out", 4 * 768);

    // 5. Residual 1
    pimi_add(res1, x, attn_out);
    CHECK("6. residual_1", res1, h_check, "res1", 4 * 768);

    // 6. LayerNorm 2
    pimi_layernorm(ln2, res1, ln2_w, ln2_b, 1e-5f);
    CHECK("7. ln2", ln2, h_check, "ln2", 4 * 768);

    // 7. MLP FC + bias
    pimi_matmul(mlp_fc, ln2, fc_w);
    pimi_add_bias(mlp_fc, fc_b);
    CHECK("8. mlp_fc", mlp_fc, h_check_mlp, "mlp_fc", 4 * 3072);

    // 8. GELU
    pimi_gelu(mlp_gelu, mlp_fc);
    CHECK("9. mlp_gelu", mlp_gelu, h_check_mlp, "mlp_gelu", 4 * 3072);

    // 9. MLP Proj + bias
    pimi_matmul(mlp_proj, mlp_gelu, proj_w);
    pimi_add_bias(mlp_proj, proj_b);
    CHECK("10. mlp_proj", mlp_proj, h_check, "mlp_proj", 4 * 768);

    // 10. Residual 2 (final block output)
    pimi_add(out, res1, mlp_proj);
    CHECK("11. block_output", out, h_check, "out", 4 * 768);

    printf("---------------------+--------------+-------\n\n");

    // Cleanup
    pimi_tensor_free(x); pimi_tensor_free(ln1); pimi_tensor_free(q); pimi_tensor_free(k); pimi_tensor_free(v);
    pimi_tensor_free(context); pimi_tensor_free(attn_out); pimi_tensor_free(res1);
    pimi_tensor_free(ln2); pimi_tensor_free(mlp_fc); pimi_tensor_free(mlp_gelu); pimi_tensor_free(mlp_proj); pimi_tensor_free(out);

    pimi_tensor_free(ln1_w); pimi_tensor_free(ln1_b); pimi_tensor_free(q_w); pimi_tensor_free(k_w); pimi_tensor_free(v_w);
    pimi_tensor_free(out_w); pimi_tensor_free(out_b);
    pimi_tensor_free(ln2_w); pimi_tensor_free(ln2_b); pimi_tensor_free(fc_w); pimi_tensor_free(fc_b);
    pimi_tensor_free(proj_w); pimi_tensor_free(proj_b);

    pimi_tensor_free(h_check); pimi_tensor_free(h_check_mlp);
    free_refs(refs, ref_count);
    pimi_shutdown();
    return 0;
}
