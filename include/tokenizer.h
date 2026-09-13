#ifndef PIMI_TOKENIZER_H
#define PIMI_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct Tokenizer Tokenizer;

Tokenizer *tokenizer_load(const char *path);
void tokenizer_free(Tokenizer *tok);

int tokenizer_encode(Tokenizer *tok, const char *text, int *out_tokens, int max_tokens);
const char *tokenizer_decode(Tokenizer *tok, int token_id);

#endif // PIMI_TOKENIZER_H
