#pragma once

#include "llama.h"
#include "ggml.h"

#include <map>
#include <memory>
#include <functional>
#include <vector>

struct llama_ubatch;
class llama_batch_allocr;
class llama_io_write_i;
class llama_io_read_i;
struct llama_context;
struct ggml_tensor;
class llama_kv_cache;
class llama_memory_recurrent;
struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// Helper functions
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);
bool llama_memory_status_is_fail(llama_memory_status status);

// Forward declarations for memory contexts
class llama_kv_cache_context;
class llama_kv_cache_iswa_context;
class llama_memory_recurrent_context;
class llama_memory_hybrid_context;

// ============================================================================
// BASE INTERFACE: llama_memory_context_i
// ============================================================================

class llama_memory_context_i {
public:
    virtual ~llama_memory_context_i() = default;
    
    // Batch processing
    virtual bool next() = 0;
    virtual bool apply() = 0;
    
    // Status and info
    virtual llama_memory_status get_status() const = 0;
    virtual const llama_ubatch & get_ubatch() const = 0;
    virtual const llama_context* get_context() const = 0;
    
    // Common methods for all memory types (with default implementations)
    virtual uint32_t get_head() const { return 0; }
    virtual uint32_t get_size() const { return 0; }
    virtual uint32_t get_n_kv() const { return 0; }
    virtual uint32_t get_n_rs() const { return 0; }
    virtual int32_t get_rs_z() const { return -1; }
    
    // Tensor access methods (for recurrent and hybrid memory)
    virtual ggml_tensor * get_k(ggml_context * ctx, int32_t il) const { return nullptr; }
    virtual ggml_tensor * get_v(ggml_context * ctx, int32_t il) const { return nullptr; }
    virtual ggml_tensor * get_r_l(int32_t il) const { return nullptr; }
    virtual ggml_tensor * get_s_l(int32_t il) const { return nullptr; }
    
    // Copy operations
    virtual ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const { return nullptr; }
    virtual ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const { return nullptr; }
    
    // Input setup
    virtual ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const { return nullptr; }
    virtual ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const { return nullptr; }
    
    virtual void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {}
    virtual void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {}
    virtual void set_input_k_shift(ggml_tensor * dst) const {}
    virtual void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {}
    virtual void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {}
    
    // Specialized accessors (return nullptr if not applicable)
    virtual const llama_kv_cache_context * as_kv_cache() const { return nullptr; }
    virtual const llama_kv_cache_iswa_context * as_kv_cache_iswa() const { return nullptr; }
    virtual const llama_memory_recurrent_context * as_recurrent() const { return nullptr; }
    virtual const llama_memory_hybrid_context * as_hybrid() const { return nullptr; }
    
    // Helper methods for graph building
    virtual int32_t s_copy(int i) const { return -1; }
    
    // Additional state info
    virtual uint32_t get_used_cells() const { return 0; }
    virtual uint32_t get_total_cells() const { return 0; }
    virtual llama_pos get_max_position() const { return 0; }
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// ============================================================================
// BASE INTERFACE: llama_memory_i
// ============================================================================

class llama_memory_i {
public:
    // Callbacks
    using layer_filter_cb = std::function<bool(int32_t il)>;
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // Factory methods
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    virtual llama_memory_context_ptr init_full() = 0;

    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // Getters
    virtual bool get_can_shift() const = 0;
    virtual uint32_t get_size() const = 0;
    virtual uint32_t get_n_seq_max() const = 0;
    
    // Access to internal components (for specialized memory types)
    virtual llama_memory_i * get_attn() const { return nullptr; }
    virtual llama_memory_i * get_recr() const { return nullptr; }
    virtual llama_kv_cache * get_kv_cache() const { return nullptr; }
    virtual llama_memory_recurrent * get_recurrent() const { return nullptr; }

    // Operations
    virtual void clear(bool data) = 0;

    // Sequence operations
    virtual bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) = 0;

    // Position info
    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    // Memory analysis
    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;
    
    // Debug and stats
    virtual uint32_t get_used_cells() const { return 0; }
    virtual uint32_t get_non_empty_cell_count() const { return 0; }
    virtual void debug_cell_states() const {}
    virtual llama_pos get_current_max_position() const { return 0; }
    
    // Reverse attention and trimming
    virtual void trim_random(int trim_percentage, const std::map<llama_pos, std::string>* token_mapping = nullptr) {}
    virtual void trim_reverse_attention_simple(int trim_percentage) {}
    virtual bool compact() { return false; }
    
    // Attention tracking
    virtual void enable_attention_tracking(bool enabled) {}
    virtual bool is_attention_tracking_enabled() const { return false; }
    virtual void set_attention_callback(
        llama_attention_callback callback,
        void* user_data) {}
    virtual void clear_attention_scores() {}
    virtual void register_attention_scores(
        int layer,
        const std::vector<float>& attention_matrix,
        size_t n_kv,
        size_t n_tokens) {}

    // State persistence
    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, 
                            llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read(llama_io_read_i & io, llama_seq_id seq_id = -1,
                           llama_state_seq_flags flags = 0) = 0;
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;

// ============================================================================
// HELPER TEMPLATES FOR SAFE CASTING
// ============================================================================

template<typename T>
T* llama_memory_as(llama_memory_i* memory) {
    return dynamic_cast<T*>(memory);
}

template<typename T>
const T* llama_memory_as(const llama_memory_i* memory) {
    return dynamic_cast<const T*>(memory);
}

template<typename T>
T* llama_memory_context_as(llama_memory_context_i* ctx) {
    return dynamic_cast<T*>(ctx);
}

template<typename T>
const T* llama_memory_context_as(const llama_memory_context_i* ctx) {
    return dynamic_cast<const T*>(ctx);
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

inline bool llama_memory_is_kv_cache(const llama_memory_i* memory) {
    // Use the virtual method instead of dynamic_cast
    return memory ? (memory->get_kv_cache() != nullptr) : false;
}

inline bool llama_memory_is_recurrent(const llama_memory_i* memory) {
    // Use the virtual method instead of dynamic_cast
    return memory ? (memory->get_recurrent() != nullptr) : false;
}

// For hybrid, we need to check if it's NOT kv_cache and NOT recurrent
inline bool llama_memory_is_hybrid(const llama_memory_i* memory) {
    if (!memory) return false;
    
    // Check if it has both attention and recurrent components
    return (memory->get_attn() != nullptr && memory->get_recr() != nullptr);
}

inline const llama_kv_cache_context* llama_memory_context_to_kv_cache(
    const llama_memory_context_i* ctx) {
    return ctx ? ctx->as_kv_cache() : nullptr;
}

inline const llama_memory_recurrent_context* llama_memory_context_to_recurrent(
    const llama_memory_context_i* ctx) {
    return ctx ? ctx->as_recurrent() : nullptr;
}

inline const llama_memory_hybrid_context* llama_memory_context_to_hybrid(
    const llama_memory_context_i* ctx) {
    return ctx ? ctx->as_hybrid() : nullptr;
}
