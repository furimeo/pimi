#ifndef CPU_OPS_H
#define CPU_OPS_H

void cpu_vec_add(const float *a, const float *b, float *c, int n);
void cpu_gemm(const float *a, const float *b, float *c, int M, int N, int K);
void cpu_layernorm(const float *x, const float *gamma, const float *beta, float *out, int rows, int cols, float eps);
void cpu_softmax(const float *x, float *out, int rows, int cols);

#endif // CPU_OPS_H
