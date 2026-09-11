#ifndef CUDA_DRV_H
#define CUDA_DRV_H

#include <windows.h>
#include <stdint.h>
#include <stddef.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef uint64_t CUdeviceptr;
typedef void* CUevent;

#define CUDA_SUCCESS 0

typedef CUresult (__stdcall *PFN_cuInit)(unsigned int Flags);
typedef CUresult (__stdcall *PFN_cuDeviceGetCount)(int *count);
typedef CUresult (__stdcall *PFN_cuDeviceGet)(CUdevice *device, int ordinal);
typedef CUresult (__stdcall *PFN_cuDeviceGetName)(char *name, int len, CUdevice dev);
typedef CUresult (__stdcall *PFN_cuDeviceComputeCapability)(int *major, int *minor, CUdevice dev);
typedef CUresult (__stdcall *PFN_cuDeviceTotalMem_v2)(size_t *bytes, CUdevice dev);
typedef CUresult (__stdcall *PFN_cuCtxCreate_v2)(CUcontext *pctx, unsigned int flags, CUdevice dev);
typedef CUresult (__stdcall *PFN_cuCtxDestroy_v2)(CUcontext ctx);
typedef CUresult (__stdcall *PFN_cuCtxSynchronize)(void);
typedef CUresult (__stdcall *PFN_cuMemAlloc_v2)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (__stdcall *PFN_cuMemFree_v2)(CUdeviceptr dptr);
typedef CUresult (__stdcall *PFN_cuMemcpyHtoD_v2)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
typedef CUresult (__stdcall *PFN_cuMemcpyDtoH_v2)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult (__stdcall *PFN_cuModuleLoadData)(CUmodule *module, const void *image);
typedef CUresult (__stdcall *PFN_cuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);
typedef CUresult (__stdcall *PFN_cuLaunchKernel)(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, void *hStream,
    void **kernelParams, void **extra
);
typedef CUresult (__stdcall *PFN_cuEventCreate)(CUevent *phEvent, unsigned int Flags);
typedef CUresult (__stdcall *PFN_cuEventDestroy_v2)(CUevent hEvent);
typedef CUresult (__stdcall *PFN_cuEventRecord)(CUevent hEvent, void *hStream);
typedef CUresult (__stdcall *PFN_cuEventSynchronize)(CUevent hEvent);
typedef CUresult (__stdcall *PFN_cuEventElapsedTime)(float *pMilliseconds, CUevent hStart, CUevent hEnd);
typedef CUresult (__stdcall *PFN_cuMemGetInfo_v2)(size_t *free, size_t *total);

typedef struct {
    HMODULE hLib;
    CUdevice dev;
    CUcontext ctx;
    char devName[128];
    int ccMajor;
    int ccMinor;
    size_t totalMem;
    size_t freeMem;

    // Function pointers
    PFN_cuInit cuInit;
    PFN_cuDeviceGetCount cuDeviceGetCount;
    PFN_cuDeviceGet cuDeviceGet;
    PFN_cuDeviceGetName cuDeviceGetName;
    PFN_cuDeviceComputeCapability cuDeviceComputeCapability;
    PFN_cuDeviceTotalMem_v2 cuDeviceTotalMem_v2;
    PFN_cuCtxCreate_v2 cuCtxCreate_v2;
    PFN_cuCtxDestroy_v2 cuCtxDestroy_v2;
    PFN_cuCtxSynchronize cuCtxSynchronize;
    PFN_cuMemAlloc_v2 cuMemAlloc_v2;
    PFN_cuMemFree_v2 cuMemFree_v2;
    PFN_cuMemcpyHtoD_v2 cuMemcpyHtoD_v2;
    PFN_cuMemcpyDtoH_v2 cuMemcpyDtoH_v2;
    PFN_cuModuleLoadData cuModuleLoadData;
    PFN_cuModuleGetFunction cuModuleGetFunction;
    PFN_cuLaunchKernel cuLaunchKernel;
    PFN_cuEventCreate cuEventCreate;
    PFN_cuEventDestroy_v2 cuEventDestroy_v2;
    PFN_cuEventRecord cuEventRecord;
    PFN_cuEventSynchronize cuEventSynchronize;
    PFN_cuEventElapsedTime cuEventElapsedTime;
    PFN_cuMemGetInfo_v2 cuMemGetInfo_v2;
} CudaContext;

int cuda_init(CudaContext *ctx);
void cuda_cleanup(CudaContext *ctx);

#endif // CUDA_DRV_H
