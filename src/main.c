#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>
#include "pimi.h"
#include "model.h"
#include "tokenizer.h"

#define MAX_SEQ_LEN 512

typedef struct {
    int id;
    float prob;
} ProbIndex;

static int compare_prob(const void *a, const void *b) {
    float diff = ((const ProbIndex*)b)->prob - ((const ProbIndex*)a)->prob;
    if (diff > 0.0f) return 1;
    if (diff < 0.0f) return -1;
    return 0;
}

static int sample_token(float *logits, int vocab_size, const int *context_tokens, int context_len, float temp, float top_p, float repeat_penalty) {
    // 1. Repetition penalty
    if (repeat_penalty != 1.0f && context_tokens && context_len > 0) {
        for (int i = 0; i < context_len; ++i) {
            int tid = context_tokens[i];
            if (tid >= 0 && tid < vocab_size) {
                if (logits[tid] > 0.0f) logits[tid] /= repeat_penalty;
                else logits[tid] *= repeat_penalty;
            }
        }
    }

    if (temp <= 0.0f) {
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }
        return best;
    }

    // Scale by temperature
    float inv_temp = 1.0f / temp;
    float max_l = logits[0] * inv_temp;
    for (int i = 1; i < vocab_size; ++i) {
        float l = logits[i] * inv_temp;
        if (l > max_l) max_l = l;
    }

    // Softmax
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        float e = expf(logits[i] * inv_temp - max_l);
        logits[i] = e;
        sum += e;
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_sum;
    }

    // Top-k pre-filter (take top 64 candidates for fast top-p sorting)
    ProbIndex top_candidates[64];
    for (int k = 0; k < 64; ++k) {
        top_candidates[k].id = -1;
        top_candidates[k].prob = -1.0f;
    }

    for (int i = 0; i < vocab_size; ++i) {
        float p = logits[i];
        if (p > top_candidates[63].prob) {
            top_candidates[63].prob = p;
            top_candidates[63].id = i;
            // bubble up
            for (int k = 62; k >= 0; --k) {
                if (top_candidates[k + 1].prob > top_candidates[k].prob) {
                    ProbIndex tmp = top_candidates[k];
                    top_candidates[k] = top_candidates[k + 1];
                    top_candidates[k + 1] = tmp;
                } else break;
            }
        }
    }

    // Top-p truncation
    float cumsum = 0.0f;
    int num_kept = 0;
    for (int k = 0; k < 64 && top_candidates[k].id >= 0; ++k) {
        cumsum += top_candidates[k].prob;
        num_kept++;
        if (cumsum >= top_p) break;
    }

    // Renormalize and sample
    float r = ((float)rand() / (float)RAND_MAX) * cumsum;
    float acc = 0.0f;
    for (int k = 0; k < num_kept; ++k) {
        acc += top_candidates[k].prob;
        if (r <= acc) return top_candidates[k].id;
    }
    return top_candidates[0].id;
}

static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}

int main(int argc, char **argv) {
    const char *model_path = "models/tinystories_33m.pimi";
    const char *tok_path   = "models/tokenizer.bin";
    const char *ptx_path   = "src/kernels.ptx";
    const char *prompt     = "Once upon a time";
    int max_new_tokens     = 64;
    float temp             = 0.7f;
    float top_p            = 0.9f;
    float repeat_penalty   = 1.1f;
    unsigned int seed      = 42;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) tok_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) ptx_path = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) max_new_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) repeat_penalty = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (unsigned int)atoi(argv[++i]);
        else if (argv[i][0] != '-') prompt = argv[i];
    }


    srand(seed);

    // Check file paths with fallback
    FILE *ft = fopen(tok_path, "rb");
    if (!ft) {
        tok_path = "../models/tokenizer.bin";
        ft = fopen(tok_path, "rb");
    }
    if (ft) fclose(ft);

    FILE *fm = fopen(model_path, "rb");
    if (!fm) {
        model_path = "../models/tinystories_33m.pimi";
        fm = fopen(model_path, "rb");
    }
    if (fm) fclose(fm);

    FILE *fp = fopen(ptx_path, "rb");
    if (!fp) {
        ptx_path = "../src/kernels.ptx";
        fp = fopen(ptx_path, "rb");
    }
    if (fp) fclose(fp);

    printf("loading tokenizer: %s\n", tok_path);
    Tokenizer *tok = tokenizer_load(tok_path);
    if (!tok) {
        fprintf(stderr, "error: failed to load tokenizer (%s)\n", tok_path);
        return 1;
    }

    printf("loading model: %s\n", model_path);
    GptNeoModel *model = model_load(model_path, ptx_path, MAX_SEQ_LEN);
    if (!model) {
        fprintf(stderr, "error: failed to load model (%s)\n", model_path);
        tokenizer_free(tok);
        return 1;
    }

    char unescaped_prompt[2048];
    int uj = 0;
    for (int ui = 0; prompt[ui] && uj < 2047; ++ui) {
        if (prompt[ui] == '\\' && prompt[ui + 1] == 'n') {
            unescaped_prompt[uj++] = '\n';
            ui++;
        } else {
            unescaped_prompt[uj++] = prompt[ui];
        }
    }
    unescaped_prompt[uj] = '\0';

    int tokens[MAX_SEQ_LEN];
    int prompt_len = tokenizer_encode(tok, unescaped_prompt, tokens, MAX_SEQ_LEN);
    if (prompt_len <= 0) {
        fprintf(stderr, "error: failed to tokenize prompt: \"%s\"\n", prompt);
        model_free(model);
        tokenizer_free(tok);
        return 1;
    }

    printf("\n--- generation ---\n");
    printf("%s", unescaped_prompt);
    fflush(stdout);

    int cur_len = prompt_len;
    int gen_count = 0;
    double t_start = get_time_ms();

    while (gen_count < max_new_tokens && cur_len < MAX_SEQ_LEN) {
        float *logits = model_forward(model, tokens, cur_len);
        if (!logits) break;

        int next_tok = sample_token(logits, model_vocab_size(model), tokens, cur_len, temp, top_p, repeat_penalty);
        if (next_tok == 50256) break; // EOS

        const char *piece = tokenizer_decode(tok, next_tok);
        printf("%s", piece);
        fflush(stdout);

        tokens[cur_len++] = next_tok;
        gen_count++;
    }



    double t_total = get_time_ms() - t_start;
    printf("\n--- done ---\n");
    printf("\ngenerated %d tokens in %.1f ms (%.2f tok/s)\n",
           gen_count, t_total, (gen_count > 0) ? (gen_count / (t_total / 1000.0)) : 0.0);

    model_free(model);
    tokenizer_free(tok);
    return 0;
}
