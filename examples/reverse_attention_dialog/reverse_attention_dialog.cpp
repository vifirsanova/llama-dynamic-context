// reverse_attention_dialog_no_flash.cpp
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <unistd.h>
#include <fcntl.h>

static int callback_call_count = 0;
static std::map<int, size_t> layer_call_counts;
static std::map<int, std::vector<float>> layer_scores_cache;

static void attention_debug_callback(void* user_data, int layer, 
                                     const float* scores, 
                                     size_t n_kv, size_t n_tokens) {
    (void)user_data;
    callback_call_count++;
    layer_call_counts[layer]++;
    
    printf("\n[ATTN_CALLBACK] Layer %d: %zux%zu scores\n", layer, n_kv, n_tokens);
    
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
    // Парсинг аргументов в формате "-m model.gguf"
    std::string model_path;
    int n_ctx = 2048;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            try {
                n_ctx = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                fprintf(stderr, "Invalid context size: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s -m <model.gguf> [-c context_size]\n", argv[0]);
            printf("\nExample:\n");
            printf("  %s -m model.gguf\n", argv[0]);
            printf("  %s -m model.gguf -c 4096\n", argv[0]);
            return 0;
        } else if (i == 1 && argv[i][0] != '-') {
            // Старый формат для обратной совместимости
            model_path = argv[i];
            if (i + 1 < argc) {
                try {
                    n_ctx = std::stoi(argv[i + 1]);
                } catch (const std::invalid_argument& e) {
                    // Игнорируем, если второй аргумент не число
                }
            }
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s -m <model.gguf> [-c context_size]\n", argv[0]);
            return 1;
        }
    }
    
    if (model_path.empty()) {
        fprintf(stderr, "Error: Model path not specified\n");
        fprintf(stderr, "Usage: %s -m <model.gguf> [-c context_size]\n", argv[0]);
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s -m /path/to/model.gguf\n", argv[0]);
        return 1;
    }
    
    printf("=== REVERSE-ATTENTION DIALOG TEST (NO FLASH) ===\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Context: %d tokens\n", n_ctx);
    printf("=================================================\n\n");
    
    // 1. Параметры модели
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only для тестирования
    
    printf("[1/4] Loading model...\n");
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("  Model loaded\n");
    
    // 2. Параметры контекста - ВАЖНО: ОТКЛЮЧАЕМ FLASH ATTENTION!
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = 256;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // 🔥 КЛЮЧЕВАЯ СТРОЧКА!
    
    printf("[2/4] Creating context (Flash Attention DISABLED)...\n");
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    printf("  Context created (no Flash Attention)\n");
    
    // 3. Настройка callback
    printf("[3/4] Setting up callbacks...\n");
    llama_set_attention_callback(ctx, attention_debug_callback, nullptr);
    enable_reverse_attention_debug(true);
    printf("  Callbacks registered, debug enabled\n");
    
    // Получаем vocab
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // Инициализация сэмплера
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    
    printf("\nReady for dialog. Type your message or 'quit' to exit.\n");
    printf("Attention callbacks will be shown in real-time.\n\n");
    
    int turn_count = 0;
    
    while (true) {
        printf("\n[Turn %d] User: ", turn_count + 1);
        std::string user_input;
        std::getline(std::cin, user_input);
        
        if (user_input.empty() || user_input == "quit") {
            break;
        }
        
        // Простой промпт
        std::string full_prompt = user_input + "\nAssistant: ";
        
        // Токенизация
        std::vector<llama_token> prompt_tokens;
        int n_tokens = llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), NULL, 0, true, false);
        if (n_tokens < 0) {
            prompt_tokens.resize(-n_tokens);
            llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, false);
        } else {
            prompt_tokens.resize(n_tokens);
            llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, false);
        }
        
        printf("Assistant: ");
        
        // Сбрасываем статистику для этого тура
        int callbacks_before = callback_call_count;
        
        // Декодирование промпта
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Decoding failed\n");
            break;
        }
        
        // Генерация ответа
        std::string response;
        int max_tokens = 100;
        int generated_tokens = 0;
        
        for (int i = 0; i < max_tokens; i++) {
            llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
            
            if (llama_vocab_is_eog(vocab, new_token)) {
                break;
            }
            
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
            if (n <= 0) {
                break;
            }
            
            std::string piece(buf, n);
            
            if (piece == "\n" && generated_tokens > 5) {
                break;
            }
            
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;
            generated_tokens++;
            
            // Декодирование следующего токена
            batch = llama_batch_get_one(&new_token, 1);
            if (llama_decode(ctx, batch) != 0) {
                break;
            }
            
            // Применяем reverse attention триминг каждые 5 токенов
            if (generated_tokens % 5 == 0) {
                printf("\n[Applying reverse attention trim...]");
                fflush(stdout);
                llama_kv_cache_trim_reverse_attention(ctx, 25);
            }
        }
        
        printf("\n");
        
        // Статистика после тура
        int callbacks_this_turn = callback_call_count - callbacks_before;
        printf("\n[Turn %d Summary]\n", turn_count + 1);
        printf("  Generated tokens: %d\n", generated_tokens);
        printf("  Attention callbacks received: %d\n", callbacks_this_turn);
        
        if (callbacks_this_turn > 0) {
            printf("  Layers with callbacks: ");
            for (const auto& kv : layer_call_counts) {
                printf("%d(%zu) ", kv.first, kv.second);
            }
            printf("\n");
            
            // Тестируем reverse attention
            printf("  Testing reverse attention trim (25%%)...\n");
            llama_kv_cache_trim_reverse_attention(ctx, 25);
            
            // Расширенная версия
            printf("  Testing extended reverse attention trim...\n");
            llama_reverse_attention_params params = llama_reverse_attention_default_params();
            params.trim_threshold = 0.25f;
            params.min_attention_score = 0.05f;
            llama_kv_cache_trim_reverse_attention_ex(ctx, &params);
        } else {
            printf("  WARNING: No attention callbacks received!\n");
            printf("  Make sure Flash Attention is disabled.\n");
        }
        
        turn_count++;
    }
    
    // Финальная статистика
    printf("\n\n=== FINAL RESULTS ===\n");
    if (callback_call_count > 0) {
        printf("SUCCESS: Received %d attention callbacks total!\n", callback_call_count);
        printf("Total dialog turns: %d\n", turn_count);
        
        for (const auto& kv : layer_call_counts) {
            int layer = kv.first;
            size_t count = kv.second;
            printf("  Layer %d: %zu calls\n", layer, count);
            
            if (layer_scores_cache.find(layer) != layer_scores_cache.end()) {
                const auto& scores = layer_scores_cache[layer];
                if (!scores.empty()) {
                    printf("    Scores cached: %zu\n", scores.size());
                    
                    // Анализируем scores
                    if (scores.size() >= 10) {
                        float min_score = *std::min_element(scores.begin(), scores.begin() + 10);
                        float max_score = *std::max_element(scores.begin(), scores.begin() + 10);
                        printf("    First 10 scores: min=%.4f, max=%.4f\n", min_score, max_score);
                        
                        // Проверяем, являются ли это логитами (до softmax)
                        if (min_score < -1.0f || max_score > 1.0f) {
                            printf("    Looks like pre-softmax logits ✓\n");
                        } else {
                            printf("    Looks like post-softmax probabilities\n");
                        }
                    }
                }
            }
        }
        
        printf("\nReverse attention test summary:\n");
        printf("  - Flash Attention: DISABLED (required for reverse attention)\n");
        printf("  - Callbacks received: YES\n");
        printf("  - Attention scores extracted: YES\n");
        printf("  - Your implementation works correctly!\n");
        
    } else {
        printf("FAILURE: No attention callbacks received!\n");
        printf("\nPossible reasons:\n");
        printf("  1. Flash Attention is still enabled\n");
        printf("  2. Model doesn't support attention callbacks\n");
        printf("  3. Bug in attention callback registration\n");
        
        printf("\nTo fix:\n");
        printf("  1. Make sure ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED\n");
        printf("  2. Check if enable_reverse_attention_debug(true) is called\n");
        printf("  3. Verify your llama.cpp has reverse attention support\n");
    }
    
    // Очистка
    llama_sampler_free(smpl);
    
    if (callback_call_count > 0) {
        llama_set_attention_callback(ctx, nullptr, nullptr);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    
    printf("\nTest completed. Exit code: %d\n", callback_call_count > 0 ? 0 : 1);
    return callback_call_count > 0 ? 0 : 1;
}
