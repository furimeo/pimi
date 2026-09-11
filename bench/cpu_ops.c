#include "cpu_ops.h"
#include <math.h>
#include <string.h>

void cpu_vec_add(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Cache-friendly i-k-j loop order for CPU GEMM
void cpu_gemm(const float *a, const float *b, float *c, int M, int N, int K) {
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

void cpu_layernorm(const float *x, const float *gamma, const float *beta, float *out, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float *x_r = &x[r * cols];
        float *out_r = &out[r * cols];

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += x_r[c];
        }
        float mean = sum / (float)cols;

        float var_sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float diff = x_r[c] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / (float)cols;
        float inv_std = 1.0f / sqrtf(var + eps);

        for (int c = 0; c < cols; ++c) {
            out_r[c] = ((x_r[c] - mean) * inv_std) * gamma[c] + beta[c];
        }
    }
}

void cpu_softmax(const float *x, float *out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float *x_r = &x[r * cols];
        float *out_r = &out[r * cols];

        float max_val = x_r[0];
        for (int c = 1; c < cols; ++c) {
            if (x_r[c] > max_val) max_val = x_r[c];
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float exp_val = expf(x_r[c] - max_val);
            out_r[c] = exp_val;
            sum += exp_val;
        }

        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; ++c) {
            out_r[c] *= inv_sum;
        }
    }
}
