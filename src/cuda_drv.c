#include "cuda_drv.h"
#include <stdio.h>

#define LOAD(name) \
    ctx->name = (PFN_##name)GetProcAddress(ctx->hLib, #name); \
    if (!ctx->name) { \
        fprintf(stderr, "cuda_drv: missing symbol %s\n", #name); \
        return -1; \
    }

int cuda_drv_init(CudaContext *ctx) {
    memset(ctx, 0, sizeof(CudaContext));

    ctx->hLib = LoadLibraryA("nvcuda.dll");
    if (!ctx->hLib) {
        fprintf(stderr, "cuda_drv: nvcuda.dll not found\n");
        return -1;
    }

    LOAD(cuInit);
    LOAD(cuDeviceGetCount);
    LOAD(cuDeviceGet);
    LOAD(cuDeviceGetName);
    LOAD(cuDeviceComputeCapability);
    LOAD(cuDeviceTotalMem_v2);
    LOAD(cuCtxCreate_v2);
    LOAD(cuCtxDestroy_v2);
    LOAD(cuCtxSynchronize);
    LOAD(cuMemAlloc_v2);
    LOAD(cuMemFree_v2);
    LOAD(cuMemcpyHtoD_v2);
    LOAD(cuMemcpyDtoH_v2);
    LOAD(cuMemcpyDtoD_v2);
    LOAD(cuMemsetD8_v2);
    LOAD(cuModuleLoadData);
    LOAD(cuModuleGetFunction);
    LOAD(cuLaunchKernel);
    LOAD(cuMemGetInfo_v2);

    if (ctx->cuInit(0) != CUDA_SUCCESS) {
        cuda_drv_cleanup(ctx);
        return -2;
    }

    int dev_count = 0;
    if (ctx->cuDeviceGetCount(&dev_count) != CUDA_SUCCESS || dev_count <= 0) {
        cuda_drv_cleanup(ctx);
        return -3;
    }

    ctx->cuDeviceGet(&ctx->dev, 0);
    ctx->cuDeviceGetName(ctx->devName, sizeof(ctx->devName), ctx->dev);
    ctx->cuDeviceComputeCapability(&ctx->ccMajor, &ctx->ccMinor, ctx->dev);
    ctx->cuDeviceTotalMem_v2(&ctx->totalMem, ctx->dev);

    if (ctx->cuCtxCreate_v2(&ctx->ctx, 0, ctx->dev) != CUDA_SUCCESS) {
        cuda_drv_cleanup(ctx);
        return -4;
    }

    size_t free_b = 0, total_b = 0;
    ctx->cuMemGetInfo_v2(&free_b, &total_b);
    ctx->freeMem = free_b;

    return 0;
}


void cuda_drv_cleanup(CudaContext *ctx) {
    if (ctx->ctx) {
        ctx->cuCtxDestroy_v2(ctx->ctx);
        ctx->ctx = NULL;
    }
    if (ctx->hLib) {
        FreeLibrary(ctx->hLib);
        ctx->hLib = NULL;
    }
}
