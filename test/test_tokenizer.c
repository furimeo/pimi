#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

int main(void) {
    printf("testing pimi tokenizer...\n");

    const char *tok_path = "models/tokenizer.bin";
    Tokenizer *tok = tokenizer_load(tok_path);
    if (!tok) {
        tok_path = "../models/tokenizer.bin";
        tok = tokenizer_load(tok_path);
    }
    if (!tok) {
        fprintf(stderr, "error: failed to load tokenizer.bin\n");
        return 1;
    }

    int tokens[128];

    // Test 1
    const char *p1 = "Once upon a time";
    int n1 = tokenizer_encode(tok, p1, tokens, 128);
    printf("Prompt 1: \"%s\" (%d tokens)\n  Tokens: ", p1, n1);
    for (int i = 0; i < n1; ++i) printf("%d ", tokens[i]);
    printf("\n  Decoded: \"");
    for (int i = 0; i < n1; ++i) printf("%s", tokenizer_decode(tok, tokens[i]));
    printf("\"\n");

    int expected1[] = {7454, 2402, 257, 640};
    int match1 = (n1 == 4);
    for (int i = 0; i < 4 && match1; ++i) {
        if (tokens[i] != expected1[i]) match1 = 0;
    }
    printf("  Test 1 match: %s\n\n", match1 ? "PASS" : "FAIL");

    // Test 2
    const char *p2 = "Once upon a time, there was a little girl named Lily.";
    int n2 = tokenizer_encode(tok, p2, tokens, 128);
    printf("Prompt 2: \"%s\" (%d tokens)\n  Tokens: ", p2, n2);
    for (int i = 0; i < n2; ++i) printf("%d ", tokens[i]);
    printf("\n  Decoded: \"");
    for (int i = 0; i < n2; ++i) printf("%s", tokenizer_decode(tok, tokens[i]));
    printf("\"\n");

    int expected2[] = {7454, 2402, 257, 640, 11, 612, 373, 257, 1310, 2576, 3706, 20037, 13};
    int match2 = (n2 == 13);
    for (int i = 0; i < 13 && match2; ++i) {
        if (tokens[i] != expected2[i]) match2 = 0;
    }
    printf("  Test 2 match: %s\n\n", match2 ? "PASS" : "FAIL");

    tokenizer_free(tok);
    if (!match1 || !match2) return 1;
    printf("All tokenizer tests passed!\n");
    return 0;
}
