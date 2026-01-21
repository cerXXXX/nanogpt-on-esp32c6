#include "inference.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_random.h" // Нужно для генерации случайных чисел

static const char* TAG = "INFERENCE";

// --- Ссылки на встроенные файлы (из CMakeLists.txt) ---
// Внимание: точки в именах файлов меняются на подчеркивания
extern const uint8_t model_bin_start[] asm("_binary_stories1M_bin_start");
extern const uint8_t tokenizer_bin_start[] asm("_binary_tokenizer_bin_start");
extern const uint8_t tokenizer_bin_end[]   asm("_binary_tokenizer_bin_end");

// --- Вспомогательные математические функции ---

void rmsnorm(float* o, float* x, const float* weight, int size) {
    // 1. Считаем сумму квадратов
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // 2. Нормализуем и умножаем на вес
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // Находим max для стабильности (чтобы не было переполнения exp)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    // Exp и сумма
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // Нормализация
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, const float* w, int n, int d) {
    // W (d, n) @ x (n) -> xout (d)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const float* w_row = w + i * n;
        for (int j = 0; j < n; j++) {
            val += w_row[j] * x[j];
        }
        xout[i] = val;
    }
}

// --- Инициализация и память ---

void build_transformer(Transformer* t) {
    ESP_LOGI(TAG, "Loading model info...");

    const int* ptr = (const int*)model_bin_start;
    t->config.dim = ptr[0];
    t->config.hidden_dim = ptr[1];
    t->config.n_layers = ptr[2];
    t->config.n_heads = ptr[3];
    t->config.n_kv_heads = ptr[4];
    t->config.vocab_size = ptr[5];
    t->config.seq_len = ptr[6];

    // !!! ВАЖНОЕ ИЗМЕНЕНИЕ !!!
    // У вас 16 слоев. Это очень много для RAM.
    // 64 токена = 262 КБ (Не влезает вместе с WiFi).
    // 32 токена = 131 КБ (Должно влезть).
    int max_seq_len_ram = 32; 
    
    if (t->config.seq_len > max_seq_len_ram) {
        t->config.seq_len = max_seq_len_ram; 
        ESP_LOGW(TAG, "Context limited to %d tokens to save RAM", max_seq_len_ram);
    }

    Config* p = &t->config;
    
    // Смещения весов
    const float* w = (const float*)(model_bin_start + 7 * sizeof(int));
    int head_size = p->dim / p->n_heads;
    
    t->token_embedding_table = w; w += p->vocab_size * p->dim;
    t->rms_att_weight = w; w += p->n_layers * p->dim;
    t->wq = w; w += p->n_layers * p->dim * (p->n_heads * head_size);
    t->wk = w; w += p->n_layers * p->dim * (p->n_kv_heads * head_size);
    t->wv = w; w += p->n_layers * p->dim * (p->n_kv_heads * head_size);
    t->wo = w; w += p->n_layers * (p->n_heads * head_size) * p->dim;
    t->rms_ffn_weight = w; w += p->n_layers * p->dim;
    t->w1 = w; w += p->n_layers * p->hidden_dim * p->dim;
    t->w2 = w; w += p->n_layers * p->dim * p->hidden_dim;
    t->w3 = w; w += p->n_layers * p->hidden_dim * p->dim;
    t->rms_final_weight = w; 
    
    // Аллокация RAM
    ESP_LOGI(TAG, "Allocating RAM...");
    
    // Пробуем выделить память. Если не выйдет - выводим свободную память.
    t->state.x = calloc(p->dim, sizeof(float));
    t->state.xb = calloc(p->dim, sizeof(float));
    t->state.xb2 = calloc(p->dim, sizeof(float));
    t->state.hb = calloc(p->hidden_dim, sizeof(float));
    t->state.hb2 = calloc(p->hidden_dim, sizeof(float));
    t->state.logits = calloc(p->vocab_size, sizeof(float));

    // KV Cache
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_size = p->n_layers * p->seq_len * kv_dim;
    
    ESP_LOGI(TAG, "KV Cache target: %d bytes. Heap free: %d bytes", 
             (int)(kv_size * sizeof(float) * 2), 
             (int)esp_get_free_heap_size());
    
    t->state.key_cache = calloc(kv_size, sizeof(float));
    t->state.value_cache = calloc(kv_size, sizeof(float));

    if (!t->state.key_cache || !t->state.value_cache || !t->state.logits) {
        ESP_LOGE(TAG, "CRITICAL: Malloc failed! Try reducing seq_len even more.");
        abort();
    }

    ESP_LOGI(TAG, "Model Ready.");
}

void free_transformer(Transformer* t) {
    free(t->state.x); 
    free(t->state.xb); 
    free(t->state.xb2);
    free(t->state.hb); 
    free(t->state.hb2); 
    free(t->state.logits);
    free(t->state.key_cache); 
    free(t->state.value_cache);
}

// --- FORWARD PASS (Инференс) ---

float* forward(Transformer* t, int token, int pos) {
    Config* p = &t->config;
    RunState* s = &t->state;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;

    // 1. Копируем Embedding токена
    memcpy(s->x, t->token_embedding_table + token * dim, dim * sizeof(float));

    // Цикл по слоям
    for(int l = 0; l < p->n_layers; l++) {
        
        // --- Attention Block ---
        
        // Pre-norm
        rmsnorm(s->xb, s->x, t->rms_att_weight + l * dim, dim);

        // QKV Matrix Multiplications
        int layer_offset_q = l * dim * dim;
        int layer_offset_kv = l * dim * kv_dim;
        int layer_offset_o = l * dim * dim;

        // Q -> xb2
        matmul(s->xb2, s->xb, t->wq + layer_offset_q, dim, dim);
        // K -> hb
        matmul(s->hb, s->xb, t->wk + layer_offset_kv, dim, kv_dim);
        // V -> hb2
        matmul(s->hb2, s->xb, t->wv + layer_offset_kv, dim, kv_dim);

        // --- RoPE (Исправленная версия) ---
        for (int i = 0; i < dim; i+=2) {
            // Частота зависит от позиции внутри "головы" (head), а не глобального индекса
            // i % head_size дает индекс внутри текущей головы (0..head_size-1)
            int val_idx = i % head_size;
            
            float freq_arg = (float)val_idx / (float)head_size;
            float freq = 1.0f / powf(10000.0f, freq_arg);
            
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);

            // Вращаем Q
            float q0 = s->xb2[i];
            float q1 = s->xb2[i+1];
            s->xb2[i]   = q0 * fcr - q1 * fci;
            s->xb2[i+1] = q0 * fci + q1 * fcr;

            // Вращаем K (если индекс входит в размерность ключей)
            if (i < kv_dim) {
                float k0 = s->hb[i];
                float k1 = s->hb[i+1];
                s->hb[i]   = k0 * fcr - k1 * fci;
                s->hb[i+1] = k0 * fci + k1 * fcr;
            }
        }

        // --- Обновляем KV Cache ---
        int loff = l * p->seq_len * kv_dim;
        float* k_cache_row = s->key_cache + loff + pos * kv_dim;
        float* v_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(k_cache_row, s->hb, kv_dim * sizeof(float));
        memcpy(v_cache_row, s->hb2, kv_dim * sizeof(float));

        // --- Multi-Head Attention ---
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->xb2 + h * head_size;
            float* xb_out_head = s->xb + h * head_size;
            memset(xb_out_head, 0, head_size * sizeof(float));

            // Считаем attention scores (Q * K)
            for (int t_pos = 0; t_pos <= pos; t_pos++) {
                float* k = s->key_cache + loff + t_pos * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                s->hb[t_pos] = score; // Временно храним score в hb
            }

            // Softmax
            softmax(s->hb, pos + 1);

            // Взвешенная сумма (Score * V)
            for (int t_pos = 0; t_pos <= pos; t_pos++) {
                float* v = s->value_cache + loff + t_pos * kv_dim + (h / kv_mul) * head_size;
                float a = s->hb[t_pos];
                for (int i = 0; i < head_size; i++) {
                    xb_out_head[i] += a * v[i];
                }
            }
        }

        // Output Projection (WO)
        matmul(s->xb2, s->xb, t->wo + layer_offset_o, dim, dim);

        // Residual Connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }

        // --- Feed Forward Block (FFN) ---
        
        // Pre-norm
        rmsnorm(s->xb, s->x, t->rms_ffn_weight + l * dim, dim);

        int ffn_offset = l * hidden_dim * dim;
        
        // w1 (Gate)
        matmul(s->hb, s->xb, t->w1 + ffn_offset, dim, hidden_dim);
        // w3 (Up)
        matmul(s->hb2, s->xb, t->w3 + ffn_offset, dim, hidden_dim);

        // SwiGLU: val = silu(gate) * up
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // SiLU approximation: x / (1 + exp(-x))
            val *= (1.0f / (1.0f + expf(-val)));
            s->hb[i] = val * s->hb2[i];
        }
        
        // w2 (Down)
        matmul(s->xb, s->hb, t->w2 + ffn_offset, hidden_dim, dim);

        // Residual Connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }

    // Final Norm
    rmsnorm(s->x, s->x, t->rms_final_weight, dim);

    // Classifier (получаем логиты)
    matmul(s->logits, s->x, t->token_embedding_table, dim, p->vocab_size);

    return s->logits;
}

// --- Выборка с Температурой ---
int sample(Transformer* t, float* logits) {
    float temperature = 0.8f; 

    // 1. Применяем температуру
    for (int i = 0; i < t->config.vocab_size; i++) {
        logits[i] /= temperature;
    }

    // 2. Softmax (превращаем в вероятности)
    softmax(logits, t->config.vocab_size);

    // 3. Случайный выбор
    float r = (float)esp_random() / (float)UINT32_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < t->config.vocab_size; i++) {
        cdf += logits[i];
        if (r < cdf) {
            return i;
        }
    }
    return t->config.vocab_size - 1;
}

// --- Безопасный Декодер (Zero-RAM + Bounds Check) ---
char* decode(Transformer* t, int prev_token, int token) {
    static char buf[128]; 
    
    const uint8_t* ptr = tokenizer_bin_start;
    const uint8_t* end = tokenizer_bin_end;
    
    // ВАЖНО: Пропускаем глобальный заголовок файла (4 байта)
    ptr += sizeof(int); 
    
    // Защита: если файл битый или пустой
    if (ptr >= end) return "";

    for (int i = 0; i < t->config.vocab_size; i++) {
        // Проверка: есть ли место для чтения score + len
        if (ptr + sizeof(float) + sizeof(int) > end) return "";

        ptr += sizeof(float); // пропускаем score
        
        int len;
        memcpy(&len, ptr, sizeof(int));
        ptr += sizeof(int);

        // Валидация длины
        if (len < 0 || len > 127) return ""; // Ошибка данных
        if (ptr + len > end) return "";      // Выход за границы

        if (i == token) {
            memcpy(buf, ptr, len);
            buf[len] = '\0';
            return buf;
        }
        
        ptr += len; // Переход к следующему слову
    }
    return "";
}