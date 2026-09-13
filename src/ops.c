#include "pimi.h"
#include "cuda_drv.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

extern CudaContext g_cuda;
extern CUfunction g_fn_vec_add;
extern CUfunction g_fn_gemm;
extern CUfunction g_fn_add_bias;
extern CUfunction g_fn_gelu;
extern CUfunction g_fn_layernorm;

// CPU reference kernels
static void cpu_gemm_raw(float *c, const float *a, const float *b, int M, int N, int K) {
    memset(c, 0, sizeof(float) * M * N);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_ik = a[i * K + k];
            const float *b_k = &b[k * N];
            float *c_i = &c[i * N];
            for (int j = 0; j < N; ++j) {
                c_i[j] += a_ik * b_k[j];
            }
        }
    }
}

static void cpu_add_raw(float *c, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

static void cpu_add_bias_raw(float *x, const float *bias, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float *xr = &x[r * cols];
        for (int c = 0; c < cols; ++c) {
            xr[c] += bias[c];
        }
    }
}

static void cpu_gelu_raw(float *out, const float *x, size_t n) {
    const float s = 0.79788456f; // sqrt(2/pi)
    for (size_t i = 0; i < n; ++i) {
        float xi = x[i];
        float y = s * (xi + 0.044715f * xi * xi * xi);
        out[i] = 0.5f * xi * (1.0f + tanhf(y));
    }
}

static void cpu_layernorm_raw(float *out, const float *x, const float *gamma, const float *beta,
                              int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float *xr = &x[r * cols];
        float *outr = &out[r * cols];
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) sum += xr[c];
        float mean = sum / (float)cols;

        float var_sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float d = xr[c] - mean;
            var_sum += d * d;
        }
        float inv_std = 1.0f / sqrtf((var_sum / (float)cols) + eps);

        for (int c = 0; c < cols; ++c) {
            outr[c] = ((xr[c] - mean) * inv_std) * gamma[c] + beta[c];
        }
    }
}

static void cpu_attn_causal_raw(float *out, const float *q, const float *k, const float *v,
                                int seq_len, int hidden_dim, int num_heads) {
    int head_dim = hidden_dim / num_heads;
    // GPT-Neo attention does not scale by sqrt(head_dim)
    float scale = 1.0f;


    // Q, K, V are [seq_len, hidden_dim] where hidden_dim = num_heads * head_dim
    // Indexing: tensor[s, h * head_dim + d]
    float *scores = (float*)malloc(seq_len * sizeof(float));

    for (int h = 0; h < num_heads; ++h) {
        int h_off = h * head_dim;

        for (int i = 0; i < seq_len; ++i) {
            const float *qi = &q[i * hidden_dim + h_off];

            // 1. Calculate QK^T with causal mask
            float max_score = -1e30f;
            for (int j = 0; j <= i; ++j) {
                const float *kj = &k[j * hidden_dim + h_off];
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += qi[d] * kj[d];
                }
                float sc = dot * scale;
                scores[j] = sc;
                if (sc > max_score) max_score = sc;
            }

            // 2. Softmax over j <= i
            float sum_exp = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float exp_val = expf(scores[j] - max_score);
                scores[j] = exp_val;
                sum_exp += exp_val;
            }
            float inv_sum = 1.0f / sum_exp;
            for (int j = 0; j <= i; ++j) {
                scores[j] *= inv_sum;
            }

            // 3. Context = sum(attn_weight * V)
            float *out_i = &out[i * hidden_dim + h_off];
            for (int d = 0; d < head_dim; ++d) {
                float ctx = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    const float *vj = &v[j * hidden_dim + h_off];
                    ctx += scores[j] * vj[d];
                }
                out_i[d] = ctx;
            }
        }
    }
    free(scores);
}

int pimi_matmul(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) return -1;
    if (a->device != b->device || a->device != out->device) {
        fprintf(stderr, "pimi: matmul device mismatch\n");
        return -2;
    }

    int M = a->dims[0];
    int K = a->dims[1];
    int N = b->dims[1];
    if (b->dims[0] != K || out->dims[0] != M || out->dims[1] != N) {
        fprintf(stderr, "pimi: matmul shape mismatch ([%d,%d] x [%d,%d] != [%d,%d])\n",
                a->dims[0], a->dims[1], b->dims[0], b->dims[1], out->dims[0], out->dims[1]);
        return -3;
    }

    if (a->device == PIMI_DEVICE_CPU) {
        cpu_gemm_raw((float*)out->data, (const float*)a->data, (const float*)b->data, M, N, K);
        return 0;
    }

    CUdeviceptr da = (CUdeviceptr)(uintptr_t)a->data;
    CUdeviceptr db = (CUdeviceptr)(uintptr_t)b->data;
    CUdeviceptr dc = (CUdeviceptr)(uintptr_t)out->data;
    unsigned int m_val = M, n_val = N, k_val = K;
    void *args[] = { &da, &db, &dc, &m_val, &n_val, &k_val };

    unsigned int bx = (M == 1) ? 256 : 16;
    unsigned int by = (M == 1) ? 1 : 16;
    unsigned int gx = (N + bx - 1) / bx;
    unsigned int gy = (M + by - 1) / by;

    CUresult res = g_cuda.cuLaunchKernel(g_fn_gemm, gx, gy, 1, bx, by, 1, 0, NULL, args, NULL);
    return (res == CUDA_SUCCESS) ? 0 : -4;
}

int pimi_add(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) return -1;
    if (a->device != b->device || a->device != out->device) return -2;
    if (a->numel != b->numel || a->numel != out->numel) return -3;

    if (a->device == PIMI_DEVICE_CPU) {
        cpu_add_raw((float*)out->data, (const float*)a->data, (const float*)b->data, a->numel);
        return 0;
    }

    CUdeviceptr da = (CUdeviceptr)(uintptr_t)a->data;
    CUdeviceptr db = (CUdeviceptr)(uintptr_t)b->data;
    CUdeviceptr dc = (CUdeviceptr)(uintptr_t)out->data;
    unsigned int n_val = (unsigned int)a->numel;
    void *args[] = { &da, &db, &dc, &n_val };

    unsigned int bx = 256;
    unsigned int gx = (n_val + bx - 1) / bx;
    CUresult res = g_cuda.cuLaunchKernel(g_fn_vec_add, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
    return (res == CUDA_SUCCESS) ? 0 : -4;
}

int pimi_add_bias(Tensor *out, const Tensor *bias) {
    if (!out || !bias) return -1;
    if (out->device != bias->device) return -2;

    int rows = out->dims[0];
    int cols = out->dims[1];
    if (bias->numel != (size_t)cols) {
        fprintf(stderr, "pimi: add_bias shape mismatch\n");
        return -3;
    }

    if (out->device == PIMI_DEVICE_CPU) {
        cpu_add_bias_raw((float*)out->data, (const float*)bias->data, rows, cols);
        return 0;
    }

    CUdeviceptr dx = (CUdeviceptr)(uintptr_t)out->data;
    CUdeviceptr db = (CUdeviceptr)(uintptr_t)bias->data;
    unsigned int r_val = rows, c_val = cols;
    void *args[] = { &dx, &db, &r_val, &c_val };

    unsigned int bx = 16, by = 16;
    unsigned int gx = (cols + bx - 1) / bx;
    unsigned int gy = (rows + by - 1) / by;
    CUresult res = g_cuda.cuLaunchKernel(g_fn_add_bias, gx, gy, 1, bx, by, 1, 0, NULL, args, NULL);
    return (res == CUDA_SUCCESS) ? 0 : -4;
}

int pimi_gelu(Tensor *out, const Tensor *x) {
    if (!out || !x) return -1;
    if (out->device != x->device) return -2;
    if (out->numel != x->numel) return -3;

    if (x->device == PIMI_DEVICE_CPU) {
        cpu_gelu_raw((float*)out->data, (const float*)x->data, x->numel);
        return 0;
    }

    CUdeviceptr dx = (CUdeviceptr)(uintptr_t)x->data;
    CUdeviceptr dout = (CUdeviceptr)(uintptr_t)out->data;
    unsigned int n_val = (unsigned int)x->numel;
    void *args[] = { &dx, &dout, &n_val };

    unsigned int bx = 256;
    unsigned int gx = (n_val + bx - 1) / bx;
    CUresult res = g_cuda.cuLaunchKernel(g_fn_gelu, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
    return (res == CUDA_SUCCESS) ? 0 : -4;
}

int pimi_layernorm(Tensor *out, const Tensor *x, const Tensor *gamma, const Tensor *beta, float eps) {
    if (!out || !x || !gamma || !beta) return -1;
    if (x->device != gamma->device || x->device != beta->device || x->device != out->device) return -2;

    int rows = 1;
    for (int i = 0; i < x->ndim - 1; ++i) rows *= x->dims[i];
    int cols = x->dims[x->ndim - 1];

    if (gamma->numel != (size_t)cols || beta->numel != (size_t)cols) return -3;

    if (x->device == PIMI_DEVICE_CPU) {
        cpu_layernorm_raw((float*)out->data, (const float*)x->data,
                          (const float*)gamma->data, (const float*)beta->data,
                          rows, cols, eps);
        return 0;
    }

    CUdeviceptr dx = (CUdeviceptr)(uintptr_t)x->data;
    CUdeviceptr dg = (CUdeviceptr)(uintptr_t)gamma->data;
    CUdeviceptr db = (CUdeviceptr)(uintptr_t)beta->data;
    CUdeviceptr dout = (CUdeviceptr)(uintptr_t)out->data;
    unsigned int r_val = rows, c_val = cols;
    float eps_val = eps;
    void *args[] = { &dx, &dg, &db, &dout, &r_val, &c_val, &eps_val };

    unsigned int bx = (rows < 256) ? (rows ? rows : 1) : 256;
    unsigned int gx = (rows + bx - 1) / bx;
    CUresult res = g_cuda.cuLaunchKernel(g_fn_layernorm, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
    return (res == CUDA_SUCCESS) ? 0 : -4;
}

int pimi_attn_causal(Tensor *out, const Tensor *q, const Tensor *k, const Tensor *v, int num_heads) {
    if (!out || !q || !k || !v) return -1;
    if (q->device != k->device || q->device != v->device || q->device != out->device) return -2;

    int seq_len = q->dims[0];
    int hidden_dim = q->dims[1];

    if (q->device == PIMI_DEVICE_CPU) {
        cpu_attn_causal_raw((float*)out->data, (const float*)q->data,
                            (const float*)k->data, (const float*)v->data,
                            seq_len, hidden_dim, num_heads);
        return 0;
    }

    // On GPU: copy Q,K,V to staging buffer, compute MHA causal, copy out
    // (with seq=4..128, footprint is few KB, highly accurate)
    size_t sz = (size_t)seq_len * hidden_dim * sizeof(float);
    float *h_q = (float*)malloc(sz);
    float *h_k = (float*)malloc(sz);
    float *h_v = (float*)malloc(sz);
    float *h_out = (float*)malloc(sz);

    g_cuda.cuMemcpyDtoH_v2(h_q, (CUdeviceptr)(uintptr_t)q->data, sz);
    g_cuda.cuMemcpyDtoH_v2(h_k, (CUdeviceptr)(uintptr_t)k->data, sz);
    g_cuda.cuMemcpyDtoH_v2(h_v, (CUdeviceptr)(uintptr_t)v->data, sz);

    cpu_attn_causal_raw(h_out, h_q, h_k, h_v, seq_len, hidden_dim, num_heads);

    g_cuda.cuMemcpyHtoD_v2((CUdeviceptr)(uintptr_t)out->data, h_out, sz);

    free(h_q); free(h_k); free(h_v); free(h_out);
    return 0;
}

int pimi_softmax_cpu(Tensor *out, const Tensor *x) {
    if (!out || !x || out->device != PIMI_DEVICE_CPU || x->device != PIMI_DEVICE_CPU) return -1;
    if (out->numel != x->numel) return -2;

    const float *xr = (const float*)x->data;
    float *outr = (float*)out->data;
    size_t n = x->numel;

    float mx = xr[0];
    for (size_t i = 1; i < n; ++i) if (xr[i] > mx) mx = xr[i];

    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float e = expf(xr[i] - mx);
        outr[i] = e;
        sum += e;
    }

    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; ++i) outr[i] *= inv_sum;
    return 0;
}

int pimi_sample_argmax(const Tensor *logits) {
    if (!logits || logits->device != PIMI_DEVICE_CPU) return -1;
    const float *data = (const float*)logits->data;
    size_t n = logits->numel;
    int best_idx = 0;
    float best_val = data[0];
    for (size_t i = 1; i < n; ++i) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = (int)i;
        }
    }
    return best_idx;
}
