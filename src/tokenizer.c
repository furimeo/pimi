#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "tokenizer.h"

#define HASH_SIZE 131072
#define HASH_MASK (HASH_SIZE - 1)

typedef struct {
    uint32_t id_a;
    uint32_t id_b;
    uint32_t id_merged;
    uint32_t rank;
    int occupied;
} HashEntry;

struct Tokenizer {
    uint32_t vocab_size;
    uint32_t num_merges;
    uint32_t byte_to_token[256];
    HashEntry *hash_table;
    char **vocab_strings;
    char *strings_pool;
};

static inline uint32_t hash_pair(uint32_t a, uint32_t b) {
    uint64_t k = ((uint64_t)a << 32) | b;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (uint32_t)(k & HASH_MASK);
}

static void insert_merge(Tokenizer *tok, uint32_t a, uint32_t b, uint32_t merged, uint32_t rank) {
    uint32_t idx = hash_pair(a, b);
    for (int step = 0; step < HASH_SIZE; ++step) {
        uint32_t slot = (idx + step) & HASH_MASK;
        if (!tok->hash_table[slot].occupied) {
            tok->hash_table[slot].id_a = a;
            tok->hash_table[slot].id_b = b;
            tok->hash_table[slot].id_merged = merged;
            tok->hash_table[slot].rank = rank;
            tok->hash_table[slot].occupied = 1;
            return;
        }
    }
}

static int find_merge(const Tokenizer *tok, uint32_t a, uint32_t b, uint32_t *out_merged, uint32_t *out_rank) {
    uint32_t idx = hash_pair(a, b);
    for (int step = 0; step < HASH_SIZE; ++step) {
        uint32_t slot = (idx + step) & HASH_MASK;
        if (!tok->hash_table[slot].occupied) return 0;
        if (tok->hash_table[slot].id_a == a && tok->hash_table[slot].id_b == b) {
            *out_merged = tok->hash_table[slot].id_merged;
            *out_rank = tok->hash_table[slot].rank;
            return 1;
        }
    }
    return 0;
}

Tokenizer *tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    char magic[4];
    uint32_t version, vocab_size, num_merges;

    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PTOK", 4) != 0) {
        fclose(f);
        return NULL;
    }
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version != 1) {
        fclose(f);
        return NULL;
    }
    if (fread(&vocab_size, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    if (fread(&num_merges, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    Tokenizer *tok = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    if (!tok) { fclose(f); return NULL; }

    tok->vocab_size = vocab_size;
    tok->num_merges = num_merges;
    tok->hash_table = (HashEntry*)calloc(HASH_SIZE, sizeof(HashEntry));

    if (fread(tok->byte_to_token, sizeof(uint32_t), 256, f) != 256) {
        tokenizer_free(tok);
        fclose(f);
        return NULL;
    }

    // Read merges
    for (uint32_t i = 0; i < num_merges; ++i) {
        uint32_t entry[4];
        if (fread(entry, sizeof(uint32_t), 4, f) != 4) {
            tokenizer_free(tok);
            fclose(f);
            return NULL;
        }
        insert_merge(tok, entry[0], entry[1], entry[2], entry[3]);
    }

    // Read vocab strings
    tok->vocab_strings = (char**)calloc(vocab_size, sizeof(char*));
    // First pass to determine total bytes for pool
    long strings_pos = ftell(f);
    size_t total_pool_bytes = 0;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        uint32_t len = 0;
        if (fread(&len, sizeof(uint32_t), 1, f) != 1) break;
        fseek(f, len, SEEK_CUR);
        total_pool_bytes += len + 1; // +1 for null terminator
    }

    fseek(f, strings_pos, SEEK_SET);
    tok->strings_pool = (char*)malloc(total_pool_bytes + 1);
    char *pool_ptr = tok->strings_pool;

    for (uint32_t i = 0; i < vocab_size; ++i) {
        uint32_t len = 0;
        if (fread(&len, sizeof(uint32_t), 1, f) != 1) break;
        tok->vocab_strings[i] = pool_ptr;
        if (len > 0) {
            if (fread(pool_ptr, 1, len, f) != len) break;
        }
        pool_ptr[len] = '\0';
        pool_ptr += len + 1;
    }

    fclose(f);
    return tok;
}

void tokenizer_free(Tokenizer *tok) {
    if (!tok) return;
    if (tok->hash_table) free(tok->hash_table);
    if (tok->vocab_strings) free(tok->vocab_strings);
    if (tok->strings_pool) free(tok->strings_pool);
    free(tok);
}

const char *tokenizer_decode(Tokenizer *tok, int token_id) {
    if (!tok || token_id < 0 || (uint32_t)token_id >= tok->vocab_size) {
        return "";
    }
    return tok->vocab_strings[token_id];
}

static int is_word_char(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

int tokenizer_encode(Tokenizer *tok, const char *text, int *out_tokens, int max_tokens) {
    if (!tok || !text || !out_tokens || max_tokens <= 0) return 0;

    int n = (int)strlen(text);
    int i = 0;
    int token_count = 0;

    static const char *contractions[] = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
        "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D", NULL
    };

    while (i < n && token_count < max_tokens) {
        int chunk_start = i;
        int chunk_end = i;

        // 1. Contractions
        int matched = 0;
        for (int c = 0; contractions[c]; ++c) {
            int clen = (int)strlen(contractions[c]);
            if (i + clen <= n && strncmp(&text[i], contractions[c], clen) == 0) {
                chunk_start = i;
                chunk_end = i + clen;
                i += clen;
                matched = 1;
                break;
            }
        }
        if (matched) goto process_chunk;

        // 2. Optional leading space followed by word or non-space
        chunk_start = i;
        int has_space = 0;
        if (text[i] == ' ' && i + 1 < n && !isspace((unsigned char)text[i + 1])) {
            has_space = 1;
            i++;
        }

        if (i < n && is_word_char(text[i])) {
            while (i < n && is_word_char(text[i])) i++;
            chunk_end = i;
            goto process_chunk;
        } else if (has_space && i < n && !isspace((unsigned char)text[i])) {
            while (i < n && !isspace((unsigned char)text[i]) && !is_word_char(text[i])) {
                if (text[i] == '\'') break;
                i++;
            }
            chunk_end = i;
            goto process_chunk;
        } else {
            i = chunk_start;
        }

        // 3. Punctuation / symbols without leading space
        if (!isspace((unsigned char)text[i])) {
            chunk_start = i;
            while (i < n && !isspace((unsigned char)text[i]) && !is_word_char(text[i])) {
                if (text[i] == '\'') break;
                i++;
            }
            if (i > chunk_start) {
                chunk_end = i;
                goto process_chunk;
            }
        }

        // 4. Whitespace
        if (isspace((unsigned char)text[i])) {
            chunk_start = i;
            while (i < n && isspace((unsigned char)text[i])) i++;
            chunk_end = i;
            goto process_chunk;
        }

        // Fallback: single byte
        chunk_start = i;
        chunk_end = i + 1;
        i++;

process_chunk:;
        int chunk_len = chunk_end - chunk_start;
        if (chunk_len <= 0) continue;

        // Convert chunk bytes to base token IDs
        uint32_t t_ids[512];
        int num_ids = (chunk_len < 512) ? chunk_len : 512;
        for (int k = 0; k < num_ids; ++k) {
            unsigned char b = (unsigned char)text[chunk_start + k];
            t_ids[k] = tok->byte_to_token[b];
        }

        // BPE merge loop
        while (num_ids >= 2) {
            uint32_t best_rank = 0xFFFFFFFF;
            int best_idx = -1;
            uint32_t best_merged = 0;

            for (int k = 0; k < num_ids - 1; ++k) {
                uint32_t mid, rank;
                if (find_merge(tok, t_ids[k], t_ids[k + 1], &mid, &rank)) {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = k;
                        best_merged = mid;
                    }
                }
            }

            if (best_idx == -1) break;

            t_ids[best_idx] = best_merged;
            for (int k = best_idx + 1; k < num_ids - 1; ++k) {
                t_ids[k] = t_ids[k + 1];
            }
            num_ids--;
        }

        for (int k = 0; k < num_ids && token_count < max_tokens; ++k) {
            out_tokens[token_count++] = (int)t_ids[k];
        }
    }

    return token_count;
}
