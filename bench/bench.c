#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "cuda_drv.h"
#include "cpu_ops.h"

static double timer_ms(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) {
        QueryPerformanceFrequency(&freq);
        init = 1;
    }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
}

static char* load_text(const char* path) {
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

static float max_err(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static void rand_floats(float* a, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
    }
}

int main(void) {
    CudaContext ctx;
    if (cuda_init(&ctx) != 0) {
        fprintf(stderr, "error: cuda init failed\n");
        return 1;
    }

    char* ptx = load_text("kernels.ptx");
    if (!ptx) ptx = load_text("bench/kernels.ptx");
    if (!ptx) {
        fprintf(stderr, "error: missing kernels.ptx\n");
        cuda_cleanup(&ctx);
        return 1;
    }

    CUmodule mod;
    CUresult res = ctx.cuModuleLoadData(&mod, ptx);
    free(ptx);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "error: cuModuleLoadData %d\n", res);
        cuda_cleanup(&ctx);
        return 1;
    }

    CUfunction fn_vec_add, fn_gemm, fn_ln, fn_softmax;
    ctx.cuModuleGetFunction(&fn_vec_add, mod, "vec_add");
    ctx.cuModuleGetFunction(&fn_gemm, mod, "gemm_naive");
    ctx.cuModuleGetFunction(&fn_ln, mod, "layernorm_row");
    ctx.cuModuleGetFunction(&fn_softmax, mod, "softmax_row");

    CUevent evt_0, evt_1;
    ctx.cuEventCreate(&evt_0, 0);
    ctx.cuEventCreate(&evt_1, 0);

    printf("\n%-30s | %9s | %10s | %9s | %12s | %s\n",
           "operation", "cpu (ms)", "gpu_k (ms)", "e2e (ms)", "k_spd / e2e", "ok");
    printf("-------------------------------+-----------+------------+-----------+--------------+----\n");

    // 1. GEMM
    struct { const char* name; int m, n, k, iters; } gemm_tests[] = {
        {"gemm 128x128x128", 128, 128, 128, 50},
        {"gemm 256x256x256", 256, 256, 256, 30},
        {"gemm 512x512x512", 512, 512, 512, 10},
        {"gpt_neo qkv (1x768x768)", 1, 768, 768, 100},
        {"gpt_neo mlp (1x768x3072)", 1, 3072, 768, 50},
        {"prefill (128x768x768)", 128, 768, 768, 20}
    };

    for (int i = 0; i < 6; ++i) {
        int M = gemm_tests[i].m, N = gemm_tests[i].n, K = gemm_tests[i].k, iters = gemm_tests[i].iters;
        size_t sz_a = M * K * sizeof(float), sz_b = K * N * sizeof(float), sz_c = M * N * sizeof(float);

        float *ha = (float*)malloc(sz_a), *hb = (float*)malloc(sz_b);
        float *hc_cpu = (float*)malloc(sz_c), *hc_gpu = (float*)malloc(sz_c);
        rand_floats(ha, M * K); rand_floats(hb, K * N);

        cpu_gemm(ha, hb, hc_cpu, M, N, K);
        double t0 = timer_ms();
        for (int it = 0; it < iters; ++it) cpu_gemm(ha, hb, hc_cpu, M, N, K);
        double cpu_ms = (timer_ms() - t0) / iters;

        CUdeviceptr da, db, dc;
        ctx.cuMemAlloc_v2(&da, sz_a);
        ctx.cuMemAlloc_v2(&db, sz_b);
        ctx.cuMemAlloc_v2(&dc, sz_c);

        unsigned int m_val = M, n_val = N, k_val = K;
        void* args[] = { &da, &db, &dc, &m_val, &n_val, &k_val };
        unsigned int bx = (M == 1) ? 256 : 16, by = (M == 1) ? 1 : 16;
        unsigned int gx = (N + bx - 1) / bx, gy = (M + by - 1) / by;

        ctx.cuMemcpyHtoD_v2(da, ha, sz_a);
        ctx.cuMemcpyHtoD_v2(db, hb, sz_b);
        ctx.cuLaunchKernel(fn_gemm, gx, gy, 1, bx, by, 1, 0, NULL, args, NULL);
        ctx.cuCtxSynchronize();

        ctx.cuEventRecord(evt_0, 0);
        for (int it = 0; it < iters; ++it) {
            ctx.cuLaunchKernel(fn_gemm, gx, gy, 1, bx, by, 1, 0, NULL, args, NULL);
        }
        ctx.cuEventRecord(evt_1, 0);
        ctx.cuEventSynchronize(evt_1);
        float elapsed = 0.0f;
        ctx.cuEventElapsedTime(&elapsed, evt_0, evt_1);
        double k_ms = elapsed / iters;

        t0 = timer_ms();
        for (int it = 0; it < iters; ++it) {
            ctx.cuMemcpyHtoD_v2(da, ha, sz_a);
            ctx.cuMemcpyHtoD_v2(db, hb, sz_b);
            ctx.cuLaunchKernel(fn_gemm, gx, gy, 1, bx, by, 1, 0, NULL, args, NULL);
            ctx.cuMemcpyDtoH_v2(hc_gpu, dc, sz_c);
        }
        ctx.cuCtxSynchronize();
        double e2e_ms = (timer_ms() - t0) / iters;

        int pass = max_err(hc_cpu, hc_gpu, M * N) < 1e-2f;
        printf("%-30s | %8.3f ms | %9.3f ms | %8.3f ms | %5.2fx / %4.2fx | %s\n",
               gemm_tests[i].name, cpu_ms, k_ms, e2e_ms, cpu_ms / k_ms, cpu_ms / e2e_ms, pass ? "ok" : "fail");

        ctx.cuMemFree_v2(da); ctx.cuMemFree_v2(db); ctx.cuMemFree_v2(dc);
        free(ha); free(hb); free(hc_cpu); free(hc_gpu);
    }

    // 2. VecAdd
    struct { const char* name; int n, iters; } vec_tests[] = {
        {"vec_add n=768 (residual)", 768, 100},
        {"vec_add n=65536", 65536, 50},
        {"vec_add n=1048576 (1m)", 1048576, 20}
    };

    for (int i = 0; i < 3; ++i) {
        int N = vec_tests[i].n, iters = vec_tests[i].iters;
        size_t sz = N * sizeof(float);
        float *ha = (float*)malloc(sz), *hb = (float*)malloc(sz);
        float *hc_cpu = (float*)malloc(sz), *hc_gpu = (float*)malloc(sz);
        rand_floats(ha, N); rand_floats(hb, N);

        cpu_vec_add(ha, hb, hc_cpu, N);
        double t0 = timer_ms();
        for (int it = 0; it < iters; ++it) cpu_vec_add(ha, hb, hc_cpu, N);
        double cpu_ms = (timer_ms() - t0) / iters;

        CUdeviceptr da, db, dc;
        ctx.cuMemAlloc_v2(&da, sz);
        ctx.cuMemAlloc_v2(&db, sz);
        ctx.cuMemAlloc_v2(&dc, sz);

        unsigned int n_val = N;
        void* args[] = { &da, &db, &dc, &n_val };
        unsigned int bx = 256, gx = (N + bx - 1) / bx;

        ctx.cuMemcpyHtoD_v2(da, ha, sz);
        ctx.cuMemcpyHtoD_v2(db, hb, sz);
        ctx.cuLaunchKernel(fn_vec_add, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
        ctx.cuCtxSynchronize();

        ctx.cuEventRecord(evt_0, 0);
        for (int it = 0; it < iters; ++it) {
            ctx.cuLaunchKernel(fn_vec_add, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
        }
        ctx.cuEventRecord(evt_1, 0);
        ctx.cuEventSynchronize(evt_1);
        float elapsed = 0.0f;
        ctx.cuEventElapsedTime(&elapsed, evt_0, evt_1);
        double k_ms = elapsed / iters;

        t0 = timer_ms();
        for (int it = 0; it < iters; ++it) {
            ctx.cuMemcpyHtoD_v2(da, ha, sz);
            ctx.cuMemcpyHtoD_v2(db, hb, sz);
            ctx.cuLaunchKernel(fn_vec_add, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
            ctx.cuMemcpyDtoH_v2(hc_gpu, dc, sz);
        }
        ctx.cuCtxSynchronize();
        double e2e_ms = (timer_ms() - t0) / iters;

        int pass = max_err(hc_cpu, hc_gpu, N) < 1e-4f;
        printf("%-30s | %8.3f ms | %9.3f ms | %8.3f ms | %5.2fx / %4.2fx | %s\n",
               vec_tests[i].name, cpu_ms, k_ms, e2e_ms, cpu_ms / k_ms, cpu_ms / e2e_ms, pass ? "ok" : "fail");

        ctx.cuMemFree_v2(da); ctx.cuMemFree_v2(db); ctx.cuMemFree_v2(dc);
        free(ha); free(hb); free(hc_cpu); free(hc_gpu);
    }

    // 3. LayerNorm
    struct { const char* name; int rows, cols, iters; } ln_tests[] = {
        {"layernorm (1x768 decode)", 1, 768, 100},
        {"layernorm (128x768 prefill)", 128, 768, 50}
    };

    for (int i = 0; i < 2; ++i) {
        int rows = ln_tests[i].rows, cols = ln_tests[i].cols, iters = ln_tests[i].iters;
        size_t sz_mat = rows * cols * sizeof(float), sz_vec = cols * sizeof(float);
        float *hx = (float*)malloc(sz_mat), *hg = (float*)malloc(sz_vec), *hb = (float*)malloc(sz_vec);
        float *ho_cpu = (float*)malloc(sz_mat), *ho_gpu = (float*)malloc(sz_mat);
        rand_floats(hx, rows * cols); rand_floats(hg, cols); rand_floats(hb, cols);

        cpu_layernorm(hx, hg, hb, ho_cpu, rows, cols, 1e-5f);
        double t0 = timer_ms();
        for (int it = 0; it < iters; ++it) cpu_layernorm(hx, hg, hb, ho_cpu, rows, cols, 1e-5f);
        double cpu_ms = (timer_ms() - t0) / iters;

        CUdeviceptr dx, dg, db, _do;
        ctx.cuMemAlloc_v2(&dx, sz_mat);
        ctx.cuMemAlloc_v2(&dg, sz_vec);
        ctx.cuMemAlloc_v2(&db, sz_vec);
        ctx.cuMemAlloc_v2(&_do, sz_mat);

        unsigned int r_val = rows, c_val = cols;
        float eps_val = 1e-5f;
        void* args[] = { &dx, &dg, &db, &_do, &r_val, &c_val, &eps_val };
        unsigned int bx = (rows < 256) ? (rows ? rows : 1) : 256;
        unsigned int gx = (rows + bx - 1) / bx;

        ctx.cuMemcpyHtoD_v2(dx, hx, sz_mat);
        ctx.cuMemcpyHtoD_v2(dg, hg, sz_vec);
        ctx.cuMemcpyHtoD_v2(db, hb, sz_vec);
        ctx.cuLaunchKernel(fn_ln, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
        ctx.cuCtxSynchronize();

        ctx.cuEventRecord(evt_0, 0);
        for (int it = 0; it < iters; ++it) {
            ctx.cuLaunchKernel(fn_ln, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
        }
        ctx.cuEventRecord(evt_1, 0);
        ctx.cuEventSynchronize(evt_1);
        float elapsed = 0.0f;
        ctx.cuEventElapsedTime(&elapsed, evt_0, evt_1);
        double k_ms = elapsed / iters;

        t0 = timer_ms();
        for (int it = 0; it < iters; ++it) {
            ctx.cuMemcpyHtoD_v2(dx, hx, sz_mat);
            ctx.cuLaunchKernel(fn_ln, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
            ctx.cuMemcpyDtoH_v2(ho_gpu, _do, sz_mat);
        }
        ctx.cuCtxSynchronize();
        double e2e_ms = (timer_ms() - t0) / iters;

        int pass = max_err(ho_cpu, ho_gpu, rows * cols) < 1e-2f;
        printf("%-30s | %8.3f ms | %9.3f ms | %8.3f ms | %5.2fx / %4.2fx | %s\n",
               ln_tests[i].name, cpu_ms, k_ms, e2e_ms, cpu_ms / k_ms, cpu_ms / e2e_ms, pass ? "ok" : "fail");

        ctx.cuMemFree_v2(dx); ctx.cuMemFree_v2(dg); ctx.cuMemFree_v2(db); ctx.cuMemFree_v2(_do);
        free(hx); free(hg); free(hb); free(ho_cpu); free(ho_gpu);
    }

    // 4. Softmax
    struct { const char* name; int rows, cols, iters; } sm_tests[] = {
        {"softmax (16x512 attn)", 16, 512, 50},
        {"softmax (1x50257 vocab)", 1, 50257, 20}
    };

    for (int i = 0; i < 2; ++i) {
        int rows = sm_tests[i].rows, cols = sm_tests[i].cols, iters = sm_tests[i].iters;
        size_t sz = rows * cols * sizeof(float);
        float *hx = (float*)malloc(sz), *ho_cpu = (float*)malloc(sz), *ho_gpu = (float*)malloc(sz);
        rand_floats(hx, rows * cols);

        cpu_softmax(hx, ho_cpu, rows, cols);
        double t0 = timer_ms();
        for (int it = 0; it < iters; ++it) cpu_softmax(hx, ho_cpu, rows, cols);
        double cpu_ms = (timer_ms() - t0) / iters;

        CUdeviceptr dx, _do;
        ctx.cuMemAlloc_v2(&dx, sz);
        ctx.cuMemAlloc_v2(&_do, sz);

        unsigned int r_val = rows, c_val = cols;
        void* args[] = { &dx, &_do, &r_val, &c_val };
        unsigned int bx = (rows < 256) ? (rows ? rows : 1) : 256;
        unsigned int gx = (rows + bx - 1) / bx;

        ctx.cuMemcpyHtoD_v2(dx, hx, sz);
        ctx.cuLaunchKernel(fn_softmax, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
        ctx.cuCtxSynchronize();

        ctx.cuEventRecord(evt_0, 0);
        for (int it = 0; it < iters; ++it) {
            ctx.cuLaunchKernel(fn_softmax, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
        }
        ctx.cuEventRecord(evt_1, 0);
        ctx.cuEventSynchronize(evt_1);
        float elapsed = 0.0f;
        ctx.cuEventElapsedTime(&elapsed, evt_0, evt_1);
        double k_ms = elapsed / iters;

        t0 = timer_ms();
        for (int it = 0; it < iters; ++it) {
            ctx.cuMemcpyHtoD_v2(dx, hx, sz);
            ctx.cuLaunchKernel(fn_softmax, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL);
            ctx.cuMemcpyDtoH_v2(ho_gpu, _do, sz);
        }
        ctx.cuCtxSynchronize();
        double e2e_ms = (timer_ms() - t0) / iters;

        int pass = max_err(ho_cpu, ho_gpu, rows * cols) < 1e-2f;
        printf("%-30s | %8.3f ms | %9.3f ms | %8.3f ms | %5.2fx / %4.2fx | %s\n",
               sm_tests[i].name, cpu_ms, k_ms, e2e_ms, cpu_ms / k_ms, cpu_ms / e2e_ms, pass ? "ok" : "fail");

        ctx.cuMemFree_v2(dx); ctx.cuMemFree_v2(_do);
        free(hx); free(ho_cpu); free(ho_gpu);
    }
    printf("-------------------------------+-----------+------------+-----------+--------------+----\n\n");

    ctx.cuEventDestroy_v2(evt_0);
    ctx.cuEventDestroy_v2(evt_1);
    cuda_cleanup(&ctx);
    return 0;
}
