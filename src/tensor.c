#include "pimi.h"
#include "cuda_drv.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern CudaContext g_cuda;
extern int pimi_has_cuda(void);

Tensor *pimi_tensor_new(const int *dims, int ndim, PimiDevice device) {
    if (ndim <= 0 || ndim > 4) return NULL;

    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->numel = 1;
    for (int i = 0; i < ndim; ++i) {
        t->dims[i] = dims[i];
        t->numel *= (size_t)dims[i];
    }
    t->device = device;
    t->owns_data = 1;

    size_t bytes = t->numel * sizeof(float);
    if (device == PIMI_DEVICE_CPU) {
        t->data = malloc(bytes);
        if (!t->data) { free(t); return NULL; }
    } else if (device == PIMI_DEVICE_CUDA) {
        if (!pimi_has_cuda()) {
            fprintf(stderr, "pimi: cuda requested but not initialized\n");
            free(t);
            return NULL;
        }
        CUdeviceptr dptr;
        CUresult res = g_cuda.cuMemAlloc_v2(&dptr, bytes);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "pimi: cuMemAlloc failed (%d) for %zu bytes\n", res, bytes);
            free(t);
            return NULL;
        }
        t->data = (void*)(uintptr_t)dptr;
    }

    return t;
}

Tensor *pimi_tensor_wrap(void *data, const int *dims, int ndim, PimiDevice device) {
    if (ndim <= 0 || ndim > 4 || !data) return NULL;

    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->data = data;
    t->ndim = ndim;
    t->numel = 1;
    for (int i = 0; i < ndim; ++i) {
        t->dims[i] = dims[i];
        t->numel *= (size_t)dims[i];
    }
    t->device = device;
    t->owns_data = 0;

    return t;
}

void pimi_tensor_free(Tensor *t) {
    if (!t) return;
    if (t->owns_data && t->data) {
        if (t->device == PIMI_DEVICE_CPU) {
            free(t->data);
        } else if (t->device == PIMI_DEVICE_CUDA && pimi_has_cuda()) {
            g_cuda.cuMemFree_v2((CUdeviceptr)(uintptr_t)t->data);
        }
    }
    free(t);
}

int pimi_tensor_copy(Tensor *dst, const Tensor *src) {
    if (!dst || !src || dst->numel != src->numel) {
        fprintf(stderr, "pimi: copy dimension mismatch\n");
        return -1;
    }

    size_t bytes = dst->numel * sizeof(float);

    if (dst->device == PIMI_DEVICE_CPU && src->device == PIMI_DEVICE_CPU) {
        memcpy(dst->data, src->data, bytes);
    } else if (dst->device == PIMI_DEVICE_CUDA && src->device == PIMI_DEVICE_CPU) {
        g_cuda.cuMemcpyHtoD_v2((CUdeviceptr)(uintptr_t)dst->data, src->data, bytes);
    } else if (dst->device == PIMI_DEVICE_CPU && src->device == PIMI_DEVICE_CUDA) {
        g_cuda.cuMemcpyDtoH_v2(dst->data, (CUdeviceptr)(uintptr_t)src->data, bytes);
    } else if (dst->device == PIMI_DEVICE_CUDA && src->device == PIMI_DEVICE_CUDA) {
        g_cuda.cuMemcpyDtoD_v2((CUdeviceptr)(uintptr_t)dst->data, (CUdeviceptr)(uintptr_t)src->data, bytes);
    }
    return 0;
}

int pimi_tensor_zero(Tensor *t) {
    if (!t || !t->data) return -1;
    size_t bytes = t->numel * sizeof(float);
    if (t->device == PIMI_DEVICE_CPU) {
        memset(t->data, 0, bytes);
    } else if (t->device == PIMI_DEVICE_CUDA && pimi_has_cuda()) {
        g_cuda.cuMemsetD8_v2((CUdeviceptr)(uintptr_t)t->data, 0, bytes);
    }
    return 0;
}
