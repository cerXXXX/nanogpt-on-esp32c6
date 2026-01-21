#pragma once
#include <stdint.h>

// Ссылки на встроенные бинарники (генерируются линкером)
extern const uint8_t model_bin_start[] asm("_binary_stories1M_bin_start");
extern const uint8_t model_bin_end[]   asm("_binary_stories1M_bin_end");
extern const uint8_t tokenizer_bin_start[] asm("_binary_tokenizer_bin_start");

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    // Параметры состояния (активации), хранятся в RAM
    float *x, *xb, *xb2, *hb, *hb2;
    float *key_cache, *value_cache;
    float *logits;
    int kv_dim;
    int kv_mul;
    int head_size;
} RunState;

typedef struct {
    Config config;
    RunState state;
    // Указатели на веса (хранятся во Flash, мапятся в память)
    const float *token_embedding_table;
    const float *rms_att_weight;
    const float *wq;
    const float *wk;
    const float *wv;
    const float *wo;
    const float *rms_ffn_weight;
    const float *w1;
    const float *w2;
    const float *w3;
    const float *rms_final_weight;
    // Токенизатор
    float* vocab_scores;
} Transformer;

void build_transformer(Transformer* t);
void free_transformer(Transformer* t);
float* forward(Transformer* t, int token, int pos);
char* decode(Transformer* t, int prev_token, int token);
int sample(Transformer* t, float* logits);