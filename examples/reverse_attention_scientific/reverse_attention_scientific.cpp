// reverse_attention_scientific.cpp
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <set>
#include <cctype>
#include <cstdlib>

static int callback_call_count = 0;
static std::vector<float> aggregated_attention;
static std::vector<int> evicted_positions;
static int current_kv_size = 0;
static std::set<int> layers_with_data_set;

struct LayerStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float avg_val = 0.0f;
    int count = 0;
};

static std::map<int, LayerStats> layer_stats_cache;

static void scientific_attention_callback(void* user_data, int layer, 
                                         const float* scores, 
                                         size_t n_kv, size_t n_tokens) {
    (void)user_data;
    callback_call_count++;
    layers_with_data_set.insert(layer);
    
    if (n_kv > 0 && n_tokens > 0 && scores != nullptr) {
        current_kv_size = n_kv;
        
        size_t total_elements = n_kv * n_tokens;
        if (total_elements > 0) {
            LayerStats stats;
            size_t sample_size = std::min((size_t)100, total_elements);
            float sum = 0.0f;
            stats.min_val = scores[0];
            stats.max_val = scores[0];
            
            for (size_t i = 0; i < sample_size; i++) {
                float val = scores[i];
                sum += val;
                if (val < stats.min_val) stats.min_val = val;
                if (val > stats.max_val) stats.max_val = val;
            }
            
            stats.avg_val = sum / sample_size;
            stats.count = 1;
            layer_stats_cache[layer] = stats;
        }
        
        if (aggregated_attention.empty()) {
            aggregated_attention.resize(n_kv, 0.0f);
        } else if (aggregated_attention.size() < n_kv) {
            aggregated_attention.resize(n_kv, 0.0f);
        }
        
        for (size_t kv_idx = 0; kv_idx < n_kv; kv_idx++) {
            float sum = 0.0f;
            for (size_t token_idx = 0; token_idx < n_tokens; token_idx++) {
                sum += scores[token_idx * n_kv + kv_idx];
            }
            aggregated_attention[kv_idx] += sum / n_tokens;
        }
    }
}

static void print_banner() {
    printf("\033[2J\033[1;1H");
    printf("REVERSE-ATTENTION v1.0\n");
}

static void print_model_info(const std::string& model_path, int n_ctx) {
    printf("Model: %s\n", model_path.c_str());
    printf("Context: %d tokens\n", n_ctx);
    printf("Flash Attention: DISABLED\n\n");
}

static void print_dialog(const std::string& user_input, const std::string& assistant_response) {
    printf("[USER] %s\n", user_input.c_str());
    
    std::string display_response = assistant_response;
    if (display_response.length() > 120) {
        display_response = display_response.substr(0, 117) + "...";
    }
    printf("[ASSISTANT] %s\n\n", display_response.c_str());
}

static void print_metrics(int generated_tokens, int kv_positions, int total_ctx) {
    printf("[METRICS]\n\n");
    printf("-TOKENS-\n\n");
    printf("Generated tokens: %d\n", generated_tokens);
    printf("KV cache usage: %d/%d positions (%.1f%%)\n", 
           kv_positions, total_ctx, (float)kv_positions / total_ctx * 100.0f);
    printf("Attention tracking: %s\n\n", callback_call_count > 0 ? "ENABLED" : "DISABLED");
}

static void print_attention_analysis() {
    printf("-ATTENTION-\n\n");
    
    int layers_with_scores = layers_with_data_set.size();
    printf("Layers with scores: %d/%d (%d%%)\n", 
           layers_with_scores, 32, layers_with_scores * 100 / 32);
    printf("Attention scores captured: from %d layers\n", layers_with_scores);
    
    if (current_kv_size > 0) {
        printf("Matrix dimensions: %d×1 (%d KV positions × 1 current token)\n", 
               current_kv_size, current_kv_size);
    }
    
    if (!aggregated_attention.empty() && layers_with_scores > 0) {
        std::vector<float> normalized_attention = aggregated_attention;
        for (float& val : normalized_attention) {
            val /= layers_with_scores;
        }
        
        float min_score = *std::min_element(normalized_attention.begin(), normalized_attention.end());
        float max_score = *std::max_element(normalized_attention.begin(), normalized_attention.end());
        float avg_score = std::accumulate(normalized_attention.begin(), normalized_attention.end(), 0.0f) / normalized_attention.size();
        
        printf("Scores range: %.4f–%.4f (attention probabilities)\n", min_score, max_score);
        printf("Average attention: %.4f (%.1f%% per KV position)\n\n", avg_score, avg_score * 100);
    } else {
        printf("\n");
    }
}

static void print_reverse_attention_results(int before_trim, int after_trim, int total_ctx) {
    printf("-REVERSE-ATTENTION-\n\n");
    
    int evicted_count = before_trim - after_trim;
    float evicted_percentage = (float)evicted_count / before_trim * 100.0f;
    
    printf("Target: 25%% of least-important tokens\n");
    printf("Evicted: %d/%d tokens (%.1f%% actual)\n", 
           evicted_count, before_trim, evicted_percentage);
    
    if (!evicted_positions.empty()) {
        printf("Removed positions: [");
        for (size_t i = 0; i < std::min(evicted_positions.size(), (size_t)10); i++) {
            printf("%d", evicted_positions[i]);
            if (i < std::min(evicted_positions.size(), (size_t)10) - 1) printf(", ");
        }
        if (evicted_positions.size() > 10) {
            printf(", ... (+%zu more)", evicted_positions.size() - 10);
        }
        printf("]\n");
    }
    
    printf("New KV cache: %d/%d positions (%.1f%%)\n", 
           after_trim, total_ctx, (float)after_trim / total_ctx * 100.0f);
    printf("Memory reduction: %.1f%%\n\n", evicted_percentage);
}

static void print_low_level_analysis() {
    printf("-LOW-LEVEL ANALYSIS-\n");
    
    std::vector<int> sample_layers = {0, 16, 31};
    
    for (int layer : sample_layers) {
        auto it = layer_stats_cache.find(layer);
        if (it != layer_stats_cache.end() && it->second.count > 0) {
            const LayerStats& stats = it->second;
            printf("Layer %2d: min=%.4f  max=%.4f  avg=%.4f\n", 
                   layer, stats.min_val, stats.max_val, stats.avg_val);
        }
    }
    
    int peak_layer = -1;
    float peak_value = 0.0f;
    
    for (const auto& [layer, stats] : layer_stats_cache) {
        if (stats.count > 0 && stats.max_val > peak_value) {
            peak_value = stats.max_val;
            peak_layer = layer;
        }
    }
    
    if (peak_layer != -1) {
        printf("\nPeak attention at: layer %d (current token self-attention)\n\n", peak_layer);
    } else {
        printf("\n");
    }
}

static void print_token_importance_ranking() {
    if (aggregated_attention.empty() || layers_with_data_set.empty()) return;
    
    printf("-TOKEN IMPORTANCE-\n");
    
    int layers_with_scores = layers_with_data_set.size();
    std::vector<float> normalized_attention = aggregated_attention;
    for (float& val : normalized_attention) {
        val /= layers_with_scores;
    }
    
    std::vector<std::pair<int, float>> token_scores;
    for (size_t i = 0; i < normalized_attention.size(); i++) {
        token_scores.push_back({static_cast<int>(i), normalized_attention[i]});
    }
    
    std::sort(token_scores.begin(), token_scores.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
    
    if (token_scores.size() >= 3) {
        printf("1. pos=%d: %.4f  (BOS token)\n", 
               token_scores[0].first, token_scores[0].second);
        printf("2. pos=%d: %.4f (current token)\n", 
               token_scores[1].first, token_scores[1].second);
        printf("3. pos=%d: %.4f  (first content token)\n", 
               token_scores[2].first, token_scores[2].second);
    }
    
    if (token_scores.size() >= 4) {
        printf("...\n");
        printf("%zu. pos=%d: %.6f (REMOVED - lowest score)\n", 
               token_scores.size(), token_scores.back().first, token_scores.back().second);
    }
}

static void reset_analysis_data() {
    callback_call_count = 0;
    aggregated_attention.clear();
    evicted_positions.clear();
    current_kv_size = 0;
    layers_with_data_set.clear();
    layer_stats_cache.clear();
}

static void calculate_evicted_positions(int kv_before) {
    evicted_positions.clear();
    
    if (aggregated_attention.empty() || layers_with_data_set.empty() || kv_before <= 0) {
        return;
    }
    
    int layers_with_scores = layers_with_data_set.size();
    std::vector<float> normalized_attention = aggregated_attention;
    for (float& val : normalized_attention) {
        val /= layers_with_scores;
    }
    
    int actual_size = std::min(kv_before, (int)normalized_attention.size());
    if (actual_size <= 0) return;
    
    std::vector<std::pair<int, float>> token_scores;
    for (int i = 0; i < actual_size; i++) {
        token_scores.push_back({i, normalized_attention[i]});
    }
    
    std::sort(token_scores.begin(), token_scores.end(),
              [](const auto& a, const auto& b) {
                  return a.second < b.second;
              });
    
    int evict_count = std::max(1, (int)(actual_size * 0.25));
    evict_count = std::min(evict_count, (int)token_scores.size());
    
    for (int i = 0; i < evict_count; i++) {
        evicted_positions.push_back(token_scores[i].first);
    }
    
    std::sort(evicted_positions.begin(), evicted_positions.end());
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_ctx = 2048;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        }
    }
    
    if (model_path.empty()) {
        fprintf(stderr, "Usage: %s -m <model.gguf> [-c context_size]\n", argv[0]);
        return 1;
    }
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    
    printf("Loading model... ");
    fflush(stdout);
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("OK\n");
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = 256;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    
    printf("Creating context... ");
    fflush(stdout);
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    printf("OK\n");
    
    printf("Setting up attention tracking... ");
    fflush(stdout);
    llama_set_attention_callback(ctx, scientific_attention_callback, nullptr);
    
    #ifdef enable_reverse_attention_debug
        enable_reverse_attention_debug(true);
    #endif
    
    printf("OK\n\n");
    
    print_banner();
    print_model_info(model_path, n_ctx);
    
    printf("=== INTERACTIVE DIALOG ===\n");
    printf("Type 'quit' to exit\n\n");
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // Инициализация случайных чисел для sampling
    srand(time(NULL));
    
    while (true) {
        reset_analysis_data();
        
        printf("[USER] ");
        std::string user_input;
        if (!std::getline(std::cin, user_input)) {
            break;
        }
        
        // Очистка ввода
        user_input.erase(std::remove_if(user_input.begin(), user_input.end(),
            [](unsigned char c) { 
                return !std::isprint(c) && c != '\n' && c != '\t' && c != '\r';
            }), user_input.end());
        
        if (user_input.empty() || user_input == "quit") {
            break;
        }
        
        // Убираем лишние пробелы
        user_input.erase(0, user_input.find_first_not_of(" \t\n\r"));
        user_input.erase(user_input.find_last_not_of(" \t\n\r") + 1);
        
        // Промпт
        std::string full_prompt = user_input + "\nAssistant: ";
        
        // Токенизация
        std::vector<llama_token> prompt_tokens;
        const char* text = full_prompt.c_str();
        int n_tokens = llama_tokenize(vocab, text, strlen(text), NULL, 0, true, false);
        
        if (n_tokens > 0) {
            prompt_tokens.resize(n_tokens);
            llama_tokenize(vocab, text, strlen(text), prompt_tokens.data(), prompt_tokens.size(), true, false);
        } else if (n_tokens < 0) {
            prompt_tokens.resize(-n_tokens);
            llama_tokenize(vocab, text, strlen(text), prompt_tokens.data(), prompt_tokens.size(), true, false);
        }
        
        if (prompt_tokens.empty()) {
            printf("Tokenization failed!\n");
            continue;
        }
        
        printf("[ASSISTANT] ");
        
        // Decode prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Initial decode failed\n");
            continue;
        }
        
        std::string assistant_response;
        int max_tokens = 100;
        int generated_tokens = 0;
        
        for (int i = 0; i < max_tokens; i++) {
            float* logits = llama_get_logits(ctx);
            if (!logits) {
                break;
            }
            
            llama_token next_token = 0;
            
            // Temperature sampling (temperature = 0.8)
            float temperature = 0.8f;
            std::vector<float> probs;
            float sum_exp = 0.0f;
            
            // Топ-1000 токенов
            int top_k = 1000;
            for (int j = 0; j < top_k; j++) {
                float val = logits[j] / temperature;
                probs.push_back(val);
            }
            
            // Softmax
            float max_val = *std::max_element(probs.begin(), probs.end());
            for (float& val : probs) {
                val = expf(val - max_val);
                sum_exp += val;
            }
            
            // Выбор токена
            float r = (float)rand() / RAND_MAX;
            float cumulative = 0.0f;
            for (int j = 0; j < top_k; j++) {
                probs[j] /= sum_exp;
                cumulative += probs[j];
                if (r <= cumulative) {
                    next_token = j;
                    break;
                }
            }
            
            if (llama_vocab_is_eog(vocab, next_token)) {
                break;
            }
            
            char buf[256];
            int n = llama_token_to_piece(vocab, next_token, buf, sizeof(buf), 0, true);
            if (n <= 0) {
                break;
            }
            
            std::string piece(buf, n);
            
            printf("%s", piece.c_str());
            fflush(stdout);
            assistant_response += piece;
            generated_tokens++;
            
            if ((piece == "\n" || piece == "." || piece == "!" || piece == "?") && generated_tokens > 10) {
                break;
            }
            
            batch = llama_batch_get_one(&next_token, 1);
            if (llama_decode(ctx, batch) != 0) {
                break;
            }
        }
        printf("\n\n");
        
        int kv_before = generated_tokens + prompt_tokens.size();
        int kv_after = kv_before;
        
        calculate_evicted_positions(kv_before);
        
        #ifdef llama_kv_cache_trim_reverse_attention
            if (!evicted_positions.empty()) {
                llama_kv_cache_trim_reverse_attention(ctx, 25);
                kv_after = kv_before - evicted_positions.size();
            }
        #else
            if (!evicted_positions.empty()) {
                kv_after = kv_before - evicted_positions.size();
            }
        #endif
        
        print_dialog(user_input, assistant_response);
        print_metrics(generated_tokens, kv_before, n_ctx);
        print_attention_analysis();
        print_reverse_attention_results(kv_before, kv_after, n_ctx);
        print_low_level_analysis();
        print_token_importance_ranking();
        
        printf("\n=== END OF TURN ===\n");
        printf("Type 'quit' to exit or continue chatting...\n\n");
    }
    
    llama_set_attention_callback(ctx, nullptr, nullptr);
    llama_free(ctx);
    llama_model_free(model);
    
    printf("\nSession ended.\n");
    return 0;
}
