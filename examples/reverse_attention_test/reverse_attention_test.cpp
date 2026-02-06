// reverse_attention_test.cpp
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

static int callback_call_count = 0;
static std::map<int, size_t> layer_call_counts;

static void attention_debug_callback(void* user_data, int layer, 
                                     const float* scores, 
                                     size_t n_kv, size_t n_tokens) {
    callback_call_count++;
    layer_call_counts[layer]++;
    
    printf("[ATTN_CALLBACK] Layer %d: %zux%zu scores\n", layer, n_kv, n_tokens);
    
    if (n_kv > 0 && n_tokens > 0) {
        printf("  Sample: [0]=%.4f, [1]=%.4f, [last]=%.4f\n", 
               scores[0], 
               n_kv * n_tokens > 1 ? scores[1] : 0.0f,
               scores[n_kv * n_tokens - 1]);
    }
}

static void print_test_state(const char* phase) {
    printf("\n=== TEST STATE [%s] ===\n", phase);
    printf("Total callback calls: %d\n", callback_call_count);
    for (const auto& [layer, count] : layer_call_counts) {
        printf("Layer %d calls: %zu\n", layer, count);
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
    
    printf("=== REVERSE-ATTENTION TEST (NO FLASH) ===\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Context: %d tokens\n", n_ctx);
    printf("Flash attention: DISABLED\n");
    printf("=========================================\n\n");
    
    // 1. Параметры модели
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    
    // 2. Загрузка модели
    printf("[1/5] Loading model...\n");
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    // 3. Параметры контекста - ОТКЛЮЧАЕМ FLASH ATTENTION!
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = 512;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // ← КРИТИЧНО!
    
    printf("Context params: flash_attn_type=%d\n", ctx_params.flash_attn_type);
    
    // 4. Создание контекста
    printf("[2/5] Creating context...\n");
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    
    // 5. Настройка attention tracking
    printf("[3/5] Setting up attention callback...\n");
    llama_set_attention_callback(ctx, attention_debug_callback, nullptr);
    
    print_test_state("INITIAL");
    
    // 6. Тестовый промпт
    const char* test_prompt = "Hello world. This is a longer test prompt to generate more tokens for testing attention tracking and reverse attention trimming functionality. We need at least 50 tokens to properly test the trimming features. Let's add more text here to ensure we have enough context.Hello world. This is a longer test prompt to generate more tokens for testing attention tracking and reverse attention trimming functionality. We need at least 50 tokens to properly test the trimming features. Let's add more text here to ensure we have enough contextHello world. This is a longer test prompt to generate more tokens for testing attention tracking and reverse attention trimming functionality. We need at least 50 tokens to properly test the trimming features. Let's add more text here to ensure we have enough contextHello world. This is a longer test prompt to generate more tokens for testing attention tracking and reverse attention trimming functionality. We need at least 50 tokens to properly test the trimming features. Let's add more text here to ensure we have enough context...Hello world. This is a longer test prompt to generate more tokens for testing attention tracking and reverse attention trimming functionality. We need at least 50 tokens to properly test the trimming features. Let's add more text here to ensure we have enough context.";  
    printf("\n[4/5] Testing with prompt: \"%s\"\n", test_prompt);
    
    // 7. Токенизация
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    
    int n_tokens = -llama_tokenize(vocab, test_prompt, strlen(test_prompt), NULL, 0, true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        llama_tokenize(vocab, test_prompt, strlen(test_prompt), 
                      tokens.data(), tokens.size(), true, true);
    } else {
        tokens.resize(n_tokens);
        llama_tokenize(vocab, test_prompt, strlen(test_prompt),
                      tokens.data(), tokens.size(), true, true);
    }
    
    printf("Tokenized to %zu tokens\n", tokens.size());
    if (tokens.size() < 2) {
        printf("⚠️  Prompt too short for meaningful attention\n");
    }
    
    // 8. Forward pass
    printf("\n[5/5] Forward pass (collecting attention scores)...\n");
    
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    
    // Добавим debug output до и после decode
    printf("Before llama_decode...\n");
    int decode_result = llama_decode(ctx, batch);
    printf("After llama_decode, result=%d\n", decode_result);
    
    if (decode_result != 0) {
        fprintf(stderr, "Decode failed with code: %d\n", decode_result);
    }
    
    print_test_state("AFTER_FORWARD");
    
    // 9. Проверяем callback
    printf("\n=== CALLBACK ANALYSIS ===\n");
    if (callback_call_count == 0) {
        printf("❌ NO CALLBACKS RECEIVED\n");
        printf("\nDebug checklist:\n");
        printf("1. ✅ Flash attention disabled: %d\n", ctx_params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED);
        printf("2. ❓ Is build_attn_mha() saving attention matrix?\n");
        printf("3. ❓ Is extract_attention_scores() being called?\n");
        printf("4. ❓ Is llama_internal_attention_callback() connected?\n");
        
        // Попробуем еще раз с другим batch
        printf("\nTrying second decode...\n");
        llama_batch batch2 = llama_batch_get_one(tokens.data(), tokens.size());
        llama_decode(ctx, batch2);
        print_test_state("AFTER_SECOND_DECODE");
    } else {
        printf("✅ SUCCESS: Received %d attention callbacks!\n", callback_call_count);
        
        // 10. Тестируем trim functions
        printf("\n=== TESTING TRIMMING ===\n");
        
        printf("1. Simple trim function...\n");
        llama_kv_cache_trim_reverse_attention(ctx, 30);
        
        printf("2. Extended trim function...\n");
        llama_reverse_attention_params params;
        params.trim_threshold = 0.3f;
        params.min_attention_score = 0.05f;
        params.recent_token_weight = 1.5f;
        params.system_prompt_weight = 2.0f;
        params.min_tokens_to_keep = 100;
        params.aggregate_across_layers = true;
        params.use_cumulative_score = true;
        
        llama_kv_cache_trim_reverse_attention_ex(ctx, &params);
        
        printf("✅ Trim functions called\n");
    }
    
    // 11. Очистка
    printf("\n=== CLEANUP ===\n");
    llama_set_attention_callback(ctx, nullptr, nullptr);
    llama_free(ctx);
    llama_model_free(model);
    
    printf("\n=== FINAL SUMMARY ===\n");
    printf("Model load: ✅\n");
    printf("Context create: ✅\n");
    printf("Flash attention disabled: %s\n", 
           ctx_params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED ? "✅" : "❌");
    printf("Forward pass: %s\n", decode_result == 0 ? "✅" : "❌");
    printf("Attention callbacks: %d %s\n", callback_call_count,
           callback_call_count > 0 ? "✅✅✅" : "❌");
    
    if (callback_call_count == 0) {
        printf("\n❌ ATTENTION PIPELINE BROKEN\n");
        printf("Need to debug:\n");
        printf("1. Check if cparams.flash_attn is false in llama-graph.cpp\n");
        printf("2. Add debug prints to extract_attention_scores()\n");
        printf("3. Verify llama_internal_attention_callback() is called\n");
        return 1;
    }
    
    printf("\n🎉 REVERSE-ATTENTION PIPELINE IS WORKING!\n");
    return 0;
}
