#pragma once

#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"
#include "llama-memory.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid
//

// utilizes two instances of llama_memory_i
//   the first instance is for the attention layers of the model and the second instance is for the recurrent layers

class llama_memory_hybrid : public llama_memory_i {
public:
    llama_memory_hybrid(
            std::unique_ptr<llama_memory_i> attn,
            std::unique_ptr<llama_memory_i> recr);

    ~llama_memory_hybrid() override;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;
    uint32_t get_size() const override;
    uint32_t get_n_seq_max() const override;
    
    // Access to internal components
    llama_memory_i * get_attn() const override;
    llama_memory_i * get_recr() const override;
    llama_kv_cache * get_kv_cache() const override;
    llama_memory_recurrent * get_recurrent() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // Debug and stats
    uint32_t get_used_cells() const override;
    uint32_t get_non_empty_cell_count() const override;
    void debug_cell_states() const override;
    llama_pos get_current_max_position() const override;
    
    // Reverse attention and trimming
    void trim_random(int trim_percentage, const std::map<llama_pos, std::string>* token_mapping = nullptr) override;
    void trim_reverse_attention_simple(int trim_percentage) override;
    bool compact() override;
    
    // Attention tracking
    void enable_attention_tracking(bool enabled) override;
    bool is_attention_tracking_enabled() const override;
    void set_attention_callback(
        llama_attention_callback callback,
        void* user_data) override;
    void clear_attention_scores() override;
    void register_attention_scores(
        int layer,
        const std::vector<float>& attention_matrix,
        size_t n_kv,
        size_t n_tokens) override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

private:
    std::unique_ptr<llama_memory_i> attn;
    std::unique_ptr<llama_memory_i> recr;
};

class llama_memory_hybrid_context : public llama_memory_context_i {
public:
    // used for errors
    explicit llama_memory_hybrid_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_context(llama_memory_hybrid * mem);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_context(
            llama_memory_context_ptr ctx_attn,
            llama_memory_context_ptr ctx_recr,
            std::vector<llama_ubatch> ubatches);

    ~llama_memory_hybrid_context() override;

    //
    // llama_memory_context_i
    //

    bool next() override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;
    const llama_context* get_context() const override;
    
    // Common methods
    uint32_t get_head() const override;
    uint32_t get_size() const override;
    uint32_t get_n_kv() const override;
    uint32_t get_n_rs() const override;
    int32_t get_rs_z() const override;
    uint32_t get_used_cells() const override;
    uint32_t get_total_cells() const override;
    llama_pos get_max_position() const override;
    
    // Tensor access methods
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_r_l(int32_t il) const override;
    ggml_tensor * get_s_l(int32_t il) const override;
    
    // Copy operations
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const override;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const override;
    
    // Input setup
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    
    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_k_shift(ggml_tensor * dst) const override;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const override;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    
    // Type conversion
    const llama_kv_cache_context * as_kv_cache() const override;
    const llama_kv_cache_iswa_context * as_kv_cache_iswa() const override;
    const llama_memory_recurrent_context * as_recurrent() const override;
    const llama_memory_hybrid_context * as_hybrid() const override;
    
    // Helper methods
    int32_t s_copy(int i) const override;

    //
    // llama_memory_hybrid_context specific API
    //

    const llama_memory_context_i * get_attn() const;
    const llama_memory_context_i * get_recr() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    llama_memory_context_ptr ctx_attn;
    llama_memory_context_ptr ctx_recr;

    llama_memory_status status;
};
