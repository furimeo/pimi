#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "pimi.h"

#pragma pack(push, 1)
typedef struct {
    char magic[4];
    unsigned int version;
    unsigned int num_tensors;
    unsigned long long total_weights_bytes;
    unsigned int vocab_size;
    unsigned int hidden_dim;
    unsigned int num_layers;
    unsigned int num_heads;
    unsigned int max_seq_len;
    char padding[24];
} PimiHeader;

typedef struct {
    char name[64];
    int ndim;
    int dims[4];
    unsigned long long offset;
    unsigned long long numel;
    unsigned long long bytes;
    unsigned int reserved[4];
} PimiTensorEntry;
#pragma pack(pop)

static double timer_ms(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
}

int main(void) {
    const char *path = "models/tinystories_33m.pimi";
    FILE *f = fopen(path, "rb");
    if (!f) {
        path = "../models/tinystories_33m.pimi";
        f = fopen(path, "rb");
    }
    if (!f) {
        fprintf(stderr, "cannot open tinystories_33m.pimi\n");
        return 1;
    }

    PimiHeader h;
    fread(&h, 1, sizeof(h), f);
    if (memcmp(h.magic, "PIMI", 4) != 0) {
        fprintf(stderr, "invalid magic: %.4s\n", h.magic);
        fclose(f);
        return 1;
    }

    printf("loaded .pimi header:\n");
    printf("  version: %u, tensors: %u, size: %.2f mb\n",
           h.version, h.num_tensors, (double)h.total_weights_bytes / (1024.0 * 1024.0));
    printf("  vocab: %u, hidden: %u, layers: %u, heads: %u, seq_len: %u\n",
           h.vocab_size, h.hidden_dim, h.num_layers, h.num_heads, h.max_seq_len);

    PimiTensorEntry *entries = (PimiTensorEntry*)malloc(h.num_tensors * sizeof(PimiTensorEntry));
    fread(entries, sizeof(PimiTensorEntry), h.num_tensors, f);

    printf("\nfirst 5 tensors in table:\n");
    for (unsigned int i = 0; i < (h.num_tensors < 5 ? h.num_tensors : 5); ++i) {
        printf("  [%u] %-30s | shape [%d, %d] | offset %llu | %.2f mb\n",
               i, entries[i].name, entries[i].dims[0], entries[i].dims[1],
               entries[i].offset, (double)entries[i].bytes / (1024.0 * 1024.0));
    }

    // Benchmark loading time to GPU
    pimi_init("src/kernels.ptx");

    printf("\nallocating and uploading all 56 tensors to quadro 2000 vram...\n");
    double t0 = timer_ms();

    // Allocate host buffer to read chunk
    size_t total_uploaded = 0;
    for (unsigned int i = 0; i < h.num_tensors; ++i) {
        Tensor *t_gpu = pimi_tensor_new(entries[i].dims, entries[i].ndim, PIMI_DEVICE_CUDA);
        if (!t_gpu) {
            fprintf(stderr, "gpu oom at tensor %s\n", entries[i].name);
            break;
        }

        fseek(f, (long)entries[i].offset, SEEK_SET);
        float *host_buf = (float*)malloc(entries[i].bytes);
        fread(host_buf, 1, entries[i].bytes, f);

        Tensor *t_cpu = pimi_tensor_wrap(host_buf, entries[i].dims, entries[i].ndim, PIMI_DEVICE_CPU);
        pimi_tensor_copy(t_gpu, t_cpu);

        pimi_tensor_free(t_cpu);
        free(host_buf);
        pimi_tensor_free(t_gpu);
        total_uploaded += entries[i].bytes;
    }

    double load_time = timer_ms() - t0;
    printf("upload complete: %.2f mb transferred in %.1f ms (%.2f gb/s)\n",
           (double)total_uploaded / (1024.0 * 1024.0),
           load_time,
           ((double)total_uploaded / (1024.0 * 1024.0 * 1024.0)) / (load_time / 1000.0));

    free(entries);
    fclose(f);
    pimi_shutdown();
    return 0;
}
