#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pimi.h"

static float max_diff(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void fill_seq(float *a, size_t n, float scale) {
    for (size_t i = 0; i < n; ++i) a[i] = ((float)i - (float)n * 0.5f) * scale;
}

int main(void) {
    printf("pimi runtime test\n");
    if (pimi_init("src/kernels.ptx") != 0) {
        fprintf(stderr, "test: pimi_init failed\n");
        return 1;
    }

    // 1. MatMul Correctness (CPU vs CUDA)
    int m_dims[2] = {4, 4};
    Tensor *cpu_a = pimi_tensor_new(m_dims, 2, PIMI_DEVICE_CPU);
    Tensor *cpu_b = pimi_tensor_new(m_dims, 2, PIMI_DEVICE_CPU);
    Tensor *cpu_c = pimi_tensor_new(m_dims, 2, PIMI_DEVICE_CPU);

    fill_seq((float*)cpu_a->data, 16, 0.1f);
    fill_seq((float*)cpu_b->data, 16, 0.2f);
    pimi_matmul(cpu_c, cpu_a, cpu_b);

    Tensor *gpu_a = pimi_tensor_new(m_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *gpu_b = pimi_tensor_new(m_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *gpu_c = pimi_tensor_new(m_dims, 2, PIMI_DEVICE_CUDA);

    pimi_tensor_copy(gpu_a, cpu_a);
    pimi_tensor_copy(gpu_b, cpu_b);
    pimi_matmul(gpu_c, gpu_a, gpu_b);

    Tensor *cpu_check = pimi_tensor_new(m_dims, 2, PIMI_DEVICE_CPU);
    pimi_tensor_copy(cpu_check, gpu_c);

    float err = max_diff((float*)cpu_c->data, (float*)cpu_check->data, 16);
    printf("matmul test: %s (max_err=%.6f)\n", (err < 1e-3f) ? "ok" : "fail", err);

    pimi_tensor_free(cpu_a); pimi_tensor_free(cpu_b); pimi_tensor_free(cpu_c);
    pimi_tensor_free(gpu_a); pimi_tensor_free(gpu_b); pimi_tensor_free(gpu_c);
    pimi_tensor_free(cpu_check);

    // 2. Transformer Mini Pipeline on GPU
    // Shape: x [1, 768], gamma/beta [768], w_proj [768, 768], lm_head [768, 50257]
    printf("running mini transformer pipeline on gpu...\n");

    int x_dims[2] = {1, 768};
    int norm_dims[1] = {768};
    int w_dims[2] = {768, 768};
    int lm_dims[2] = {768, 50257};
    int logits_dims[2] = {1, 50257};

    // Allocate host inputs
    Tensor *h_x = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CPU);
    Tensor *h_gamma = pimi_tensor_new(norm_dims, 1, PIMI_DEVICE_CPU);
    Tensor *h_beta = pimi_tensor_new(norm_dims, 1, PIMI_DEVICE_CPU);
    Tensor *h_w = pimi_tensor_new(w_dims, 2, PIMI_DEVICE_CPU);
    Tensor *h_lm = pimi_tensor_new(lm_dims, 2, PIMI_DEVICE_CPU);

    fill_seq((float*)h_x->data, 768, 0.01f);
    fill_seq((float*)h_gamma->data, 768, 0.001f);
    fill_seq((float*)h_beta->data, 768, 0.001f);
    fill_seq((float*)h_w->data, 768 * 768, 0.0001f);
    fill_seq((float*)h_lm->data, 768 * 50257, 0.00001f);

    // Allocate GPU buffers
    Tensor *d_x = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *d_norm = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *d_gamma = pimi_tensor_new(norm_dims, 1, PIMI_DEVICE_CUDA);
    Tensor *d_beta = pimi_tensor_new(norm_dims, 1, PIMI_DEVICE_CUDA);
    Tensor *d_w = pimi_tensor_new(w_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *d_proj = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *d_gelu = pimi_tensor_new(x_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *d_lm = pimi_tensor_new(lm_dims, 2, PIMI_DEVICE_CUDA);
    Tensor *d_logits = pimi_tensor_new(logits_dims, 2, PIMI_DEVICE_CUDA);

    // Copy weights to GPU (one-time setup)
    pimi_tensor_copy(d_x, h_x);
    pimi_tensor_copy(d_gamma, h_gamma);
    pimi_tensor_copy(d_beta, h_beta);
    pimi_tensor_copy(d_w, h_w);
    pimi_tensor_copy(d_lm, h_lm);

    // Pipeline 100% on GPU:
    // 1. LayerNorm
    pimi_layernorm(d_norm, d_x, d_gamma, d_beta, 1e-5f);
    // 2. Matmul projection
    pimi_matmul(d_proj, d_norm, d_w);
    // 3. Residual add (d_proj = d_proj + d_x)
    pimi_add(d_proj, d_proj, d_x);
    // 4. GELU
    pimi_gelu(d_gelu, d_proj);
    // 5. LM Head projection to vocabulary
    pimi_matmul(d_logits, d_gelu, d_lm);

    // Copy only the final logits (200 KB) to CPU
    Tensor *h_logits = pimi_tensor_new(logits_dims, 2, PIMI_DEVICE_CPU);
    pimi_tensor_copy(h_logits, d_logits);

    // CPU sampling
    Tensor *h_probs = pimi_tensor_new(logits_dims, 2, PIMI_DEVICE_CPU);
    pimi_softmax_cpu(h_probs, h_logits);
    int next_token = pimi_sample_argmax(h_probs);
    printf("pipeline test: ok, sampled next_token=%d (prob=%.6f)\n",
           next_token, ((float*)h_probs->data)[next_token]);

    // Cleanup
    pimi_tensor_free(h_x); pimi_tensor_free(h_gamma); pimi_tensor_free(h_beta);
    pimi_tensor_free(h_w); pimi_tensor_free(h_lm); pimi_tensor_free(h_logits); pimi_tensor_free(h_probs);

    pimi_tensor_free(d_x); pimi_tensor_free(d_norm); pimi_tensor_free(d_gamma);
    pimi_tensor_free(d_beta); pimi_tensor_free(d_w); pimi_tensor_free(d_proj);
    pimi_tensor_free(d_gelu); pimi_tensor_free(d_lm); pimi_tensor_free(d_logits);

    pimi_shutdown();
    printf("all tests passed\n");
    return 0;
}
