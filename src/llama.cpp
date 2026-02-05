#include "llama-impl.h"

#include "llama-chat.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-kv-cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <vector>
#include <map>
#include <numeric>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// interface implementation
//

const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf                     =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);
        }
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static int llama_model_load(const std::string & fname, std::vector<std::string> & splits, llama_model & model, llama_model_params & params) {
    // loading time will be recalculated after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = 0;
    time_meas tm(model.t_load_us);

    model.t_start_us = tm.t_start_us;

    try {
        llama_model_loader ml(fname, splits, params.use_mmap, params.check_tensors, params.kv_overrides, params.tensor_buft_overrides);

        ml.print_info();

        model.hparams.vocab_only = params.vocab_only;

        try {
            model.load_arch(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            model.load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            model.load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model.load_stats(ml);
        model.print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

        if (!model.load_tensors(ml)) {
            return -2;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1;
    }

    return 0;
}

static struct llama_model * llama_model_load_from_file_impl(
        const std::string & path_model,
        std::vector<std::string> & splits,
        struct llama_model_params params) {
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    llama_model * model = new llama_model(params);

    // create list of devices to use with this model
    if (params.devices) {
        for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
            model->devices.push_back(*dev);
        }
    } else {
        // default device selection

        // build list of available devices
        std::vector<ggml_backend_dev_t> gpus;
        std::vector<ggml_backend_dev_t> igpus;
        std::vector<ggml_backend_dev_t> rpc_servers;

        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:
                case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    // skip CPU backends since they are handled separately
                    break;

                case GGML_BACKEND_DEVICE_TYPE_GPU: {
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                        rpc_servers.push_back(dev);
                    } else {
                        // check if there is already a GPU with the same device id
                        ggml_backend_dev_props props;
                        ggml_backend_dev_get_props(dev, &props);
                        auto it = std::find_if(gpus.begin(), gpus.end(), [&props](ggml_backend_dev_t d) {
                            ggml_backend_dev_props d_props;
                            ggml_backend_dev_get_props(d, &d_props);
                            if (props.device_id && d_props.device_id) {
                                return strcmp(props.device_id, d_props.device_id) == 0;
                            }
                            return false;
                        });

                        if (it != gpus.end()) {
                            LLAMA_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                                    __func__,
                                    ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                                    props.device_id ? props.device_id : "unknown id",
                                    ggml_backend_dev_name(*it), ggml_backend_dev_description(*it));
                        } else {
                            gpus.push_back(dev);
                        }
                    }
                    break;
                }

                case GGML_BACKEND_DEVICE_TYPE_IGPU:
                    igpus.push_back(dev);
                    break;
            }
        }

        // add RPC servers at the front of the list to minimize network transfers
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());

        // add GPUs
        model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());

        // add integrated GPUs only if no other devices were found
        if (model->devices.empty()) {
            model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                llama_model_free(model);
                return nullptr;
            }
            ggml_backend_dev_t main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (auto * dev : model->devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
                ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                props.device_id ? props.device_id : "unknown id",
                props.memory_free/1024/1024);
    }

    const int status = llama_model_load(path_model, splits, *model, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        llama_model_free(model);
        return nullptr;
    }

    return model;
}

// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(path_model, splits, params);
}

struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(splits.front(), splits, params);
}

// Default parameters for reverse attention trimming
llama_reverse_attention_params llama_reverse_attention_default_params(void) {
    llama_reverse_attention_params params = {
        .trim_threshold = 0.25f,
        .min_attention_score = 0.01f,
        .recent_token_weight = 1.5f,
        .system_prompt_weight = 2.0f,
        .min_tokens_to_keep = 100,
        .aggregate_across_layers = true,
        .use_cumulative_score = true,
    };
    return params;
}

// Helper function to convert API params to internal params
static reverse_attention_trim_params convert_to_internal_params(
    const llama_reverse_attention_params* api_params) {
    
    reverse_attention_trim_params params;
    params.trim_threshold = api_params->trim_threshold;
    params.min_attention_score = api_params->min_attention_score;
    params.recent_token_weight = api_params->recent_token_weight;
    params.system_prompt_weight = api_params->system_prompt_weight;
    params.min_tokens_to_keep = api_params->min_tokens_to_keep;
    params.aggregate_across_layers = api_params->aggregate_across_layers;
    params.use_cumulative_score = api_params->use_cumulative_score;
    params.preserve_system_prompt = true; // Always preserve unless explicitly disabled
    params.system_prompt_end_pos = 0; // Will be detected automatically
    
    return params;
}

void llama_kv_cache_trim_random(struct llama_context * ctx, int trim_percentage) {
    if (ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: context is null\n", __func__);
        return;
    }
    
    // Direct access to memory as friend function
    auto* memory = ctx->get_memory();
    if (memory == nullptr) {
        LLAMA_LOG_ERROR("%s: memory interface is null\n", __func__);
        return;
    }
    
    auto* kv_cache = dynamic_cast<llama_kv_cache*>(memory);
    if (kv_cache == nullptr) {
        LLAMA_LOG_ERROR("%s: memory is not a KV cache implementation\n", __func__);
        return;
    }
    
    LLAMA_LOG_INFO("%s: starting random trim on context %p\n", __func__, (void*)ctx);
    
    try {
        // Perform the random trim operation
        kv_cache->trim_random(trim_percentage);
        
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("%s: exception during random trim: %s\n", __func__, e.what());
    }
}

void llama_model_save_to_file(const struct llama_model * model, const char * path_model) {
    llama_model_saver ms(*model);
    ms.add_kv_from_model();
    ms.add_tensors_from_model();
    ms.save(path_model);
}

void llama_kv_cache_compact(struct llama_context * ctx) {
    if (ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: context is null\n", __func__);
        return;
    }
    
    // Use the SAME pattern as your working trim function
    auto* memory = ctx->get_memory();
    if (memory == nullptr) {
        LLAMA_LOG_ERROR("%s: memory interface is null\n", __func__);
        return;
    }
    
    auto* kv_cache = dynamic_cast<llama_kv_cache*>(memory);
    if (kv_cache == nullptr) {
        LLAMA_LOG_ERROR("%s: memory is not a KV cache implementation\n", __func__);
        return;
    }
    
    LLAMA_LOG_INFO("%s: starting compaction on context %p\n", __func__, (void*)ctx);
    
    try {
        bool compacted = kv_cache->compact();
        if (compacted) {
            LLAMA_LOG_INFO("%s: compaction completed successfully\n", __func__);
        } else {
            LLAMA_LOG_INFO("%s: no compaction needed\n", __func__);
        }
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("%s: exception during compaction: %s\n", __func__, e.what());
    }
}

//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count) {
    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";
    if (snprintf(split_path, maxlen, SPLIT_PATH_FORMAT, path_prefix, split_no + 1, split_count)) {
        return strlen(split_path);
    }
    return 0;
}

int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count) {
    std::string str_split_path(split_path);
    char postfix[32];
    snprintf(postfix, 32, "-%05d-of-%05d.gguf", split_no + 1, split_count);
    std::string str_postfix(postfix);

    // check if split_prefix ends with postfix
    int size_prefix = str_split_path.size() - str_postfix.size();
    if (size_prefix > 0 && str_split_path.find(str_postfix, size_prefix) != std::string::npos) {
        snprintf(split_prefix, std::min((size_t) size_prefix + 1, maxlen), "%s", split_path);
        return size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}

// Public API implementations for reverse attention trimming

// Простая версия reverse attention trimming
LLAMA_API void llama_kv_cache_trim_reverse_attention(
    struct llama_context * ctx,
    int trim_percentage) {
    
    if (ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: context is null\n", __func__);
        return;
    }
    
    auto* memory = ctx->get_memory();
    if (memory == nullptr) {
        LLAMA_LOG_ERROR("%s: memory interface is null\n", __func__);
        return;
    }
    
    auto* kv_cache = dynamic_cast<llama_kv_cache*>(memory);
    if (kv_cache == nullptr) {
        LLAMA_LOG_ERROR("%s: memory is not a KV cache implementation\n", __func__);
        return;
    }
    
    LLAMA_LOG_INFO("%s: starting reverse-attention trim on context %p\n", __func__, (void*)ctx);
    
    try {
        kv_cache->trim_reverse_attention_simple(trim_percentage);
        LLAMA_LOG_INFO("%s: reverse-attention trim completed successfully\n", __func__);
        
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("%s: exception during reverse-attention trim: %s\n", __func__, e.what());
    }
}

// Расширенная версия reverse attention trimming
LLAMA_API void llama_kv_cache_trim_reverse_attention_ex(
    struct llama_context * ctx,
    const llama_reverse_attention_params * params) {
    
    if (ctx == nullptr || params == nullptr) {
        LLAMA_LOG_ERROR("%s: invalid parameters\n", __func__);
        return;
    }
    
    auto* memory = ctx->get_memory();
    if (memory == nullptr) {
        LLAMA_LOG_ERROR("%s: memory interface is null\n", __func__);
        return;
    }
    
    auto* kv_cache = dynamic_cast<llama_kv_cache*>(memory);
    if (kv_cache == nullptr) {
        LLAMA_LOG_ERROR("%s: memory is not a KV cache implementation\n", __func__);
        return;
    }
    
    LLAMA_LOG_INFO("%s: starting extended reverse-attention trim on context %p\n", __func__, (void*)ctx);
    
    try {
        kv_cache->trim_reverse_attention_ex(params, nullptr);
        LLAMA_LOG_INFO("%s: extended reverse-attention trim completed successfully\n", __func__);
        
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("%s: exception during extended reverse-attention trim: %s\n", __func__, e.what());
    }
}

// Versions with additional parameters for backward compatibility
LLAMA_API void llama_kv_cache_trim_reverse_attention_params(
    struct llama_context * ctx,
    int trim_percentage,
    float min_attention_threshold,
    bool preserve_system_prompt) {
    
    if (ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: context is null\n", __func__);
        return;
    }
    
    llama_reverse_attention_params params = llama_reverse_attention_default_params();
    params.trim_threshold = trim_percentage / 100.0f;
    params.min_attention_score = min_attention_threshold;
    params.system_prompt_weight = preserve_system_prompt ? 2.0f : 1.0f;
    
    llama_kv_cache_trim_reverse_attention_ex(ctx, &params);
}

// Включить/выключить трекинг attention
LLAMA_API void llama_enable_attention_tracking(
    struct llama_context * ctx,
    bool enabled) {
    
    if (ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: context is null\n", __func__);
        return;
    }
    
    auto* memory = ctx->get_memory();
    if (memory == nullptr) {
        LLAMA_LOG_ERROR("%s: memory interface is null\n", __func__);
        return;
    }
    
    auto* kv_cache = dynamic_cast<llama_kv_cache*>(memory);
    if (kv_cache == nullptr) {
        LLAMA_LOG_ERROR("%s: memory is not a KV cache implementation\n", __func__);
        return;
    }
    
    try {
        kv_cache->enable_attention_tracking(enabled);
        LLAMA_LOG_INFO("%s: attention tracking %s for context %p\n", 
                      __func__, enabled ? "enabled" : "disabled", (void*)ctx);
        
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("%s: exception during attention tracking setup: %s\n", __func__, e.what());
    }
}

// Проверить включен ли трекинг attention
LLAMA_API bool llama_is_attention_tracking_enabled(
    const struct llama_context * ctx) {
    
    if (ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: context is null\n", __func__);
        return false;
    }
    
    auto* memory = ctx->get_memory();
    if (memory == nullptr) {
        LLAMA_LOG_ERROR("%s: memory interface is null\n", __func__);
        return false;
    }
    
    auto* kv_cache = dynamic_cast<llama_kv_cache*>(memory);
    if (kv_cache == nullptr) {
        LLAMA_LOG_ERROR("%s: memory is not a KV cache implementation\n", __func__);
        return false;
    }
    
    try {
        return kv_cache->is_attention_tracking_enabled();
        
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("%s: exception checking attention tracking: %s\n", __func__, e.what());
        return false;
    }
}

// Получить статистику attention
// В llama.cpp, исправьте функцию llama_get_attention_statistics:
LLAMA_API llama_attention_stats llama_get_attention_statistics(
    const struct llama_context * ctx) {
    
    llama_attention_stats stats{};
    
    if (ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: context is null\n", __func__);
        return stats;
    }
    
    auto* memory = ctx->memory.get();
    if (memory == nullptr) {
        LLAMA_LOG_ERROR("%s: memory interface is null\n", __func__);
        return stats;
    }
    
    auto* kv_cache = dynamic_cast<llama_kv_cache*>(memory);
    if (kv_cache == nullptr) {
        LLAMA_LOG_ERROR("%s: memory is not a KV cache implementation\n", __func__);
        return stats;
    }
    
    try {
        // Конвертируем internal statistics в public API
        attention_statistics internal_stats = kv_cache->get_attention_statistics();
        
        // Используем только существующие поля из llama_attention_stats
        stats.avg_attention_score = internal_stats.avg_score;
        stats.min_attention_score = internal_stats.min_score;
        stats.max_attention_score = internal_stats.max_score;
        stats.tokens_trimmed = 0; // Нужно трекать отдельно
        stats.tokens_kept = internal_stats.total_tokens;
        stats.memory_reduction = 0.0f; // Нужно трекать отдельно
        
        // Если нужны дополнительные поля, добавьте их в структуру в llama.h
        // или используйте другие доступные поля
        
        LLAMA_LOG_DEBUG("%s: retrieved attention statistics for context %p\n", 
                       __func__, (void*)ctx);
        
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("%s: exception retrieving attention statistics: %s\n", __func__, e.what());
    }
    
    return stats;
}
// Set attention callback
LLAMA_API void llama_set_attention_callback(
    struct llama_context * ctx,
    llama_attention_callback callback,
    void * user_data) {
    
    if (ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: context is null\n", __func__);
        return;
    }
    
    auto* memory = ctx->get_memory();
    if (memory == nullptr) {
        LLAMA_LOG_ERROR("%s: memory interface is null\n", __func__);
        return;
    }
    
    auto* kv_cache = dynamic_cast<llama_kv_cache*>(memory);
    if (kv_cache == nullptr) {
        LLAMA_LOG_ERROR("%s: memory is not a KV cache implementation\n", __func__);
        return;
    }
    
    try {
        kv_cache->set_attention_callback(callback, user_data);
        
        if (callback == nullptr) {
            LLAMA_LOG_INFO("%s: removed attention callback for context %p\n", 
                          __func__, (void*)ctx);
        } else {
            LLAMA_LOG_INFO("%s: set attention callback for context %p\n", 
                          __func__, (void*)ctx);
        }
        
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("%s: exception setting attention callback: %s\n", __func__, e.what());
    }
}

// Clear attention scores
LLAMA_API void llama_clear_attention_scores(
    struct llama_context * ctx) {
    
    if (ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: context is null\n", __func__);
        return;
    }
    
    auto* memory = ctx->get_memory();
    if (memory == nullptr) {
        LLAMA_LOG_ERROR("%s: memory interface is null\n", __func__);
        return;
    }
    
    auto* kv_cache = dynamic_cast<llama_kv_cache*>(memory);
    if (kv_cache == nullptr) {
        LLAMA_LOG_ERROR("%s: memory is not a KV cache implementation\n", __func__);
        return;
    }
    
    try {
        kv_cache->clear_attention_scores();
        LLAMA_LOG_INFO("%s: cleared attention scores for context %p\n", 
                      __func__, (void*)ctx);
        
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("%s: exception clearing attention scores: %s\n", __func__, e.what());
    }
}

// Internal callback for graph computation
static std::map<const llama_context*, std::pair<llama_attention_callback, void*>> g_attention_callbacks;

// Internal tracking enabled flag
static std::map<const llama_context*, bool> g_attention_tracking_enabled_internal;

// Legacy function for backward compatibility
LLAMA_API void llama_internal_attention_callback(
    const llama_context * ctx,
    int layer,
    const float * attention_scores,
    size_t n_kv,
    size_t n_tokens) {
    
    if (ctx == nullptr || attention_scores == nullptr) {
        return;
    }
    
    // Check if tracking is enabled for this context
    auto tracking_it = g_attention_tracking_enabled_internal.find(ctx);
    if (tracking_it == g_attention_tracking_enabled_internal.end() || !tracking_it->second) {
        return;
    }
    
    // Check if there's a callback registered
    auto callback_it = g_attention_callbacks.find(ctx);
    if (callback_it != g_attention_callbacks.end()) {
        const auto& [callback, user_data] = callback_it->second;
        callback(user_data, layer, attention_scores, n_kv, n_tokens);
    }
    
    // Pass to KV cache for internal tracking
    auto* memory = ctx->get_memory();
    if (memory != nullptr) {
        auto* kv_cache = dynamic_cast<llama_kv_cache*>(memory);
        if (kv_cache != nullptr && kv_cache->is_attention_tracking_enabled()) {
            std::vector<float> attention_matrix(attention_scores, attention_scores + n_kv * n_tokens);
            kv_cache->register_attention_scores(layer, attention_matrix, n_kv, n_tokens);
        }
    }
}

// Clean up callbacks when context is freed
LLAMA_API void llama_internal_cleanup_attention_callbacks(const llama_context * ctx) {
    if (ctx == nullptr) {
        return;
    }
    
    g_attention_callbacks.erase(ctx);
    g_attention_tracking_enabled_internal.erase(ctx);
    
    LLAMA_LOG_DEBUG("%s: cleaned up attention callbacks for context %p\n", 
                   __func__, (void*)ctx);
}

// Helper function to be called from llama-graph.cpp when attention scores are available
LLAMA_API void llama_graph_attention_callback(
    const llama_context * ctx,
    int layer,
    const float * attention_scores,
    size_t n_kv,
    size_t n_tokens) {
    
    llama_internal_attention_callback(ctx, layer, attention_scores, n_kv, n_tokens);
}
