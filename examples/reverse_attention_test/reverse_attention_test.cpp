// reverse_attention_test.cpp - ИСПРАВЛЕННАЯ ВЕРСИЯ
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>

static int callback_call_count = 0;
static std::map<int, size_t> layer_call_counts;
static std::map<int, std::vector<float>> layer_scores_cache;

static void attention_debug_callback(void* user_data, int layer, 
                                     const float* scores, 
                                     size_t n_kv, size_t n_tokens) {
    (void)user_data;
    callback_call_count++;
    layer_call_counts[layer]++;
    
    printf("[ATTN_CALLBACK] Layer %d: %zux%zu scores\n", layer, n_kv, n_tokens);
    
    if (n_kv > 0 && n_tokens > 0 && scores != nullptr) {
        size_t total_elements = n_kv * n_tokens;
        layer_scores_cache[layer].assign(scores, scores + total_elements);
        
        // Calculate some stats
        float sum = 0.0f, min_val = scores[0], max_val = scores[0];
        for (size_t i = 0; i < std::min((size_t)100, total_elements); i++) {
            sum += scores[i];
            min_val = std::min(min_val, scores[i]);
            max_val = std::max(max_val, scores[i]);
        }
        
        printf("  Stats (first 100): min=%.4f, max=%.4f, avg=%.4f\n", 
               min_val, max_val, sum / std::min((size_t)100, total_elements));
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [context_size]\n", argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    int n_ctx = 512;
    
    if (argc > 2) n_ctx = std::stoi(argv[2]);
    
    printf("=== REVERSE-ATTENTION TEST ===\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Context: %d tokens\n", n_ctx);
    printf("================================\n\n");
    
    // 1. Параметры модели
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    
    printf("[1/4] Loading model...\n");
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("  Model loaded\n");
    
    // 2. Параметры контекста
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = 256;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    
    printf("[2/4] Creating context...\n");
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    printf("  Context created\n");
    
    // 3. Настройка callback
    printf("[3/4] Setting up callbacks...\n");
    llama_set_attention_callback(ctx, attention_debug_callback, nullptr);
    enable_reverse_attention_debug(true);
    printf("  Callbacks registered, debug enabled\n");
    
    // 4. Тестовый промпт - ИСПРАВЛЕННАЯ ВЕРСИЯ
    printf("[4/4] Running forward pass...\n");
    
    // ПРОСТОЙ СПОСОБ: Используем предопределенный токен (обычно 1 для BOS)
    // Для большинства моделей llama BOS = 1
    llama_token bos_token = 1;
    printf("  Using BOS token: %d\n", bos_token);
    
    // Альтернатива: если модель имеет специальную функцию для BOS
    // Попробуем получить vocab из модели
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab) {
        // Если vocab доступен, используем правильный BOS
        bos_token = llama_vocab_bos(vocab);
        printf("  Actual BOS token from vocab: %d\n", bos_token);
    }
    
    // Создаем batch
    llama_batch batch = llama_batch_init(1, 0, 1);
    
    batch.token[0] = bos_token;
    batch.pos[0] = 0;           // Позиция 0 для первого токена
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;    // Нужны логиты для генерации
    batch.n_tokens = 1;
    
    printf("  First decode (BOS token at pos=0)...\n");
    int decode_result = llama_decode(ctx, batch);
    printf("  Result: %d\n", decode_result);
    
    if (decode_result == 0) {
        printf("  First decode successful\n");
        
        // Получаем логиты
        float* logits = llama_get_logits(ctx);
        
        // Простая логика выбора следующего токена
        int best_token = 0;
        float best_score = -INFINITY;
        
        // Проверяем первые 100 токенов
        for (int i = 0; i < 100; i++) {
            if (logits[i] > best_score) {
                best_score = logits[i];
                best_token = i;
            }
        }
        
        printf("  Generated token %d (score: %.2f)\n", best_token, best_score);
        
        // Декодируем второй токен
        llama_batch batch2 = llama_batch_init(1, 0, 1);
        batch2.token[0] = best_token;
        batch2.pos[0] = 1;     // Позиция 1 (следующая после BOS)
        batch2.n_seq_id[0] = 1;
        batch2.seq_id[0][0] = 0;
        batch2.logits[0] = true;
        batch2.n_tokens = 1;
        
        printf("  Second decode (token %d at pos=1)...\n", best_token);
        decode_result = llama_decode(ctx, batch2);
        printf("  Result: %d\n", decode_result);
        
        llama_batch_free(batch2);
        
        // Третий токен для большего тестирования
        if (decode_result == 0) {
            float* logits2 = llama_get_logits(ctx);
            int best_token2 = 0;
            float best_score2 = -INFINITY;
            
            for (int i = 0; i < 100; i++) {
                if (logits2[i] > best_score2) {
                    best_score2 = logits2[i];
                    best_token2 = i;
                }
            }
            
            printf("  Generated token %d (score: %.2f)\n", best_token2, best_score2);
            
            llama_batch batch3 = llama_batch_init(1, 0, 1);
            batch3.token[0] = best_token2;
            batch3.pos[0] = 2;     // Позиция 2
            batch3.n_seq_id[0] = 1;
            batch3.seq_id[0][0] = 0;
            batch3.logits[0] = true;
            batch3.n_tokens = 1;
            
            printf("  Third decode (token %d at pos=2)...\n", best_token2);
            decode_result = llama_decode(ctx, batch3);
            printf("  Result: %d\n", decode_result);
            
            llama_batch_free(batch3);
        }
    } else {
        printf("  First decode failed\n");
        
        // Попробуем альтернативный подход
        printf("\n  Trying alternative approach...\n");
        
        // Сбросим контекст
        llama_free(ctx);
        ctx = llama_init_from_model(model, ctx_params);
        llama_set_attention_callback(ctx, attention_debug_callback, nullptr);
        enable_reverse_attention_debug(true);
        
        // Тест с ручным созданием batch из 2 токенов
        llama_batch alt_batch = llama_batch_init(2, 0, 1);
        
        // Два токена
        alt_batch.token[0] = bos_token;
        alt_batch.pos[0] = 0;
        alt_batch.n_seq_id[0] = 1;
        alt_batch.seq_id[0][0] = 0;
        alt_batch.logits[0] = false;  // Не нужны логиты для первого токена
        
        // Второй токен (произвольный)
        alt_batch.token[1] = 1000;
        alt_batch.pos[1] = 1;
        alt_batch.n_seq_id[1] = 1;
        alt_batch.seq_id[1][0] = 0;
        alt_batch.logits[1] = true;   // Только последний токен имеет логиты
        
        alt_batch.n_tokens = 2;
        
        printf("  Decoding 2 tokens at positions 0 and 1...\n");
        decode_result = llama_decode(ctx, alt_batch);
        printf("  Result: %d\n", decode_result);
        
        llama_batch_free(alt_batch);
    }
    
    // 5. Проверяем результаты
    printf("\n=== RESULTS ===\n");
    if (callback_call_count > 0) {
        printf("SUCCESS: Received %d attention callbacks!\n", callback_call_count);
        for (const auto& [layer, count] : layer_call_counts) {
            printf("  Layer %d: %zu calls\n", layer, count);
            
            // Показываем данные если есть
            if (layer_scores_cache.find(layer) != layer_scores_cache.end()) {
                const auto& scores = layer_scores_cache[layer];
                if (!scores.empty()) {
                    printf("    Total scores: %zu\n", scores.size());
                    
                    // Статистика
                    int nonzero_count = 0;
                    int positive_count = 0;
                    int negative_count = 0;
                    
                    for (size_t i = 0; i < std::min((size_t)100, scores.size()); i++) {
                        float score = scores[i];
                        if (fabs(score) > 0.0001f) nonzero_count++;
                        if (score > 0.0f) positive_count++;
                        if (score < 0.0f) negative_count++;
                    }
                    
                    printf("    First 100: non-zero=%d, positive=%d, negative=%d\n",
                           nonzero_count, positive_count, negative_count);
                    
                    if (!scores.empty()) {
                        printf("    First value: %.6f\n", scores[0]);
                        printf("    Last value: %.6f\n", scores.back());
                    }
                }
            }
        }
        
        // Тестируем reverse attention функции
        printf("\nTesting reverse attention functions:\n");
        
        // Простая версия
        printf("  1. llama_kv_cache_trim_reverse_attention (25%%)...\n");
        llama_kv_cache_trim_reverse_attention(ctx, 25);
        printf("     Done\n");
        
        // Расширенная версия
        printf("  2. llama_kv_cache_trim_reverse_attention_ex...\n");
        llama_reverse_attention_params params = llama_reverse_attention_default_params();
        params.trim_threshold = 0.3f;
        params.min_attention_score = 0.05f;
        llama_kv_cache_trim_reverse_attention_ex(ctx, &params);
        printf("     Done\n");
        
    } else {
        printf("NO CALLBACKS RECEIVED\n");
        
        // Проверяем включен ли трекинг
        bool tracking_enabled = llama_is_attention_tracking_enabled(ctx);
        printf("  Attention tracking enabled: %s\n", tracking_enabled ? "YES" : "NO");
        
        // Проверяем другие функции KV cache
        printf("\nTesting basic KV cache functions:\n");
        
        // Компактируем
        printf("  1. llama_kv_cache_compact...\n");
        llama_kv_cache_compact(ctx);
        printf("     Done\n");
        
        // Случайный трим
        printf("  2. llama_kv_cache_trim_random (10%%)...\n");
        llama_kv_cache_trim_random(ctx, 10);
        printf("     Done\n");
    }
    
    // 6. Очистка
    printf("\nCleaning up...\n");
    llama_batch_free(batch);
    
    // Очищаем callback если был установлен
    if (callback_call_count > 0) {
        llama_set_attention_callback(ctx, nullptr, nullptr);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    
    return callback_call_count > 0 ? 0 : 1;
}
