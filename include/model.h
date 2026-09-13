#ifndef PIMI_MODEL_H
#define PIMI_MODEL_H

#include "pimi.h"

typedef struct {
    int vocab_size;
    int hidden_dim;
    int num_layers;
    int num_heads;
    int max_seq_len;
} GptNeoConfig;

typedef struct GptNeoModel GptNeoModel;

GptNeoModel *model_load(const char *pimi_path, const char *ptx_path, int max_seq);
void         model_free(GptNeoModel *model);
int          model_vocab_size(const GptNeoModel *model);

// Forward pass over tokens[0..seq_len-1].
// Returns pointer to CPU array of [vocab_size] floats for the last position (seq_len - 1).
float *model_forward(GptNeoModel *model, const int *tokens, int seq_len);

#endif // PIMI_MODEL_H

