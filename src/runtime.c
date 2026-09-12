#include "pimi.h"
#include "cuda_drv.h"
#include <stdio.h>
#include <stdlib.h>

CudaContext g_cuda;
static int g_cuda_available = 0;
CUmodule g_module = NULL;

CUfunction g_fn_vec_add = NULL;
CUfunction g_fn_gemm = NULL;
CUfunction g_fn_add_bias = NULL;
CUfunction g_fn_gelu = NULL;
CUfunction g_fn_layernorm = NULL;

static char* read_text(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* b = (char*)malloc(sz + 1);
    if (!b) { fclose(f); return NULL; }
    fread(b, 1, sz, f);
    b[sz] = '\0';
    fclose(f);
    return b;
}

int pimi_init(const char *ptx_path) {
    if (cuda_drv_init(&g_cuda) != 0) {
        fprintf(stderr, "pimi: cuda not available, falling back to cpu only\n");
        g_cuda_available = 0;
        return 0;
    }

    g_cuda_available = 1;

    const char* paths[] = {
        ptx_path,
        "kernels.ptx",
        "src/kernels.ptx",
        "../src/kernels.ptx"
    };

    char* ptx_code = NULL;
    for (int i = 0; i < 4; ++i) {
        if (paths[i]) {
            ptx_code = read_text(paths[i]);
            if (ptx_code) break;
        }
    }

    if (!ptx_code) {
        fprintf(stderr, "pimi: cannot locate kernels.ptx\n");
        return -1;
    }

    CUresult res = g_cuda.cuModuleLoadData(&g_module, ptx_code);
    free(ptx_code);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "pimi: cuModuleLoadData failed (%d)\n", res);
        return -2;
    }

    g_cuda.cuModuleGetFunction(&g_fn_vec_add, g_module, "vec_add");
    g_cuda.cuModuleGetFunction(&g_fn_gemm, g_module, "gemm_naive");
    g_cuda.cuModuleGetFunction(&g_fn_add_bias, g_module, "add_bias");
    g_cuda.cuModuleGetFunction(&g_fn_gelu, g_module, "gelu_new");
    g_cuda.cuModuleGetFunction(&g_fn_layernorm, g_module, "layernorm_row");

    printf("pimi: initialized on %s (sm_%d%d), %zu mb free\n",
           g_cuda.devName, g_cuda.ccMajor, g_cuda.ccMinor,
           g_cuda.freeMem / (1024 * 1024));

    return 0;
}

void pimi_shutdown(void) {
    if (g_cuda_available) {
        cuda_drv_cleanup(&g_cuda);
        g_cuda_available = 0;
    }
}

int pimi_has_cuda(void) {
    return g_cuda_available;
}
