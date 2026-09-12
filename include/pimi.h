#ifndef PIMI_H
#define PIMI_H

#include <stddef.h>

typedef enum {
    PIMI_DEVICE_CPU  = 0,
    PIMI_DEVICE_CUDA = 1
} PimiDevice;

typedef struct {
    void *data;
    int dims[4];
    int ndim;
    size_t numel;
    PimiDevice device;
    int owns_data;
} Tensor;

// Runtime lifecycle
int  pimi_init(const char *ptx_path);
void pimi_shutdown(void);
int  pimi_has_cuda(void);

// Tensor lifecycle
Tensor *pimi_tensor_new(const int *dims, int ndim, PimiDevice device);
Tensor *pimi_tensor_wrap(void *data, const int *dims, int ndim, PimiDevice device);
void    pimi_tensor_free(Tensor *t);
int     pimi_tensor_copy(Tensor *dst, const Tensor *src);
int     pimi_tensor_zero(Tensor *t);

// Operations (inputs and output must share the same device)
int pimi_matmul(Tensor *out, const Tensor *a, const Tensor *b);
int pimi_add(Tensor *out, const Tensor *a, const Tensor *b);
int pimi_add_bias(Tensor *out, const Tensor *bias);
int pimi_gelu(Tensor *out, const Tensor *x);
int pimi_layernorm(Tensor *out, const Tensor *x, const Tensor *gamma, const Tensor *beta, float eps);
int pimi_attn_causal(Tensor *out, const Tensor *q, const Tensor *k, const Tensor *v, int num_heads);

// CPU Sampling
int pimi_softmax_cpu(Tensor *out, const Tensor *x);
int pimi_sample_argmax(const Tensor *logits);

#endif // PIMI_H
