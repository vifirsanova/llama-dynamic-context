#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // llama_memory_i extended methods
    bool get_can_shift() const override;
    uint32_t get_size() const override { return size; }
    uint32_t get_n_seq_max() const override { return n_seq_max; }
    llama_memory_recurrent * get_recurrent() const override { return const_cast<llama_memory_recurrent*>(this); }
    
    // Stats and debugging
    uint32_t get_used_cells() const override { return used; }
    uint32_t get_non_empty_cell_count() const override { return used; }
    void debug_cell_states() const override;
    llama_pos get_current_max_position() const override;
    
    // Attention tracking (stubs for compatibility)
    void enable_attention_tracking(bool enabled) override {}
    bool is_attention_tracking_enabled() const override { return false; }
    void set_attention_callback(llama_attention_callback callback, void* user_data) override {}
    void clear_attention_scores() override {}
    void register_attention_scores(int layer, const std::vector<float>& attention_matrix, 
                                   size_t n_kv, size_t n_tokens) override {}
    
    // Trimming (stubs for compatibility)
    void trim_random(int trim_percentage, const std::map<llama_pos, std::string>* token_mapping = nullptr) override {}
    void trim_reverse_attention_simple(int trim_percentage) override {}
    bool compact() override { return false; }

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            llama_context * ctx = nullptr);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches,
            llama_context * ctx = nullptr);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;
    const llama_context* get_context() const override;

    // llama_memory_context_i extended methods
    uint32_t get_head() const override;
    uint32_t get_size() const override;
    uint32_t get_n_rs() const override;
    int32_t get_rs_z() const override;
    uint32_t get_used_cells() const override;
    uint32_t get_total_cells() const override;
    llama_pos get_max_position() const override;
    
    // Tensor access methods
    ggml_tensor * get_r_l(int32_t il) const override;
    ggml_tensor * get_s_l(int32_t il) const override;
    
    // Copy operations (stubs)
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const override { return nullptr; }
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const override { return nullptr; }
    
    // Input setup (stubs)
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override { return nullptr; }
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override { return nullptr; }
    
    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override {}
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override {}
    void set_input_k_shift(ggml_tensor * dst) const override {}
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const override {}
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const override {}
    
    // Type conversion
    const llama_memory_recurrent_context * as_recurrent() const override { return this; }
    
    // Tensor access for KV cache (stubs)
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const override { return nullptr; }
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const override { return nullptr; }

    //
    // llama_memory_recurrent_context specific API
    //

    int32_t s_copy(int i) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;
    llama_context * ctx;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
    
    // Cache для быстрого доступа
    mutable uint32_t cached_head = 0;
    mutable uint32_t cached_size = 0;
    mutable uint32_t cached_n_rs = 0;
    mutable int32_t cached_rs_z = -1;
    mutable uint32_t cached_used = 0;
    
    void update_cache() const;
};
