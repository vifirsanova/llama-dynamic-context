#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <unordered_map>
#include <vector>
#include <map>
#include <mutex>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

// Forward declarations for structures
struct attention_score_data;
struct reverse_attention_trim_params;
struct attention_statistics;
struct llama_reverse_attention_params;

// Structure for attention statistics (нужно объявить здесь, чтобы избежать incomplete type)
struct attention_statistics {
    float avg_score;
    float min_score;
    float max_score;
    float score_variance;
    int total_tokens;
    int tokens_with_low_score;
    int tokens_with_high_score;
    
    attention_statistics()
        : avg_score(0.0f),
          min_score(0.0f),
          max_score(0.0f),
          score_variance(0.0f),
          total_tokens(0),
          tokens_with_low_score(0),
          tokens_with_high_score(0) {}
};

// Internal structure for tracking attention scores per token
struct attention_score_data {
    std::vector<float> scores;           // Attention scores from all layers/heads
    float aggregated_score;              // Final importance score
    llama_pos position;                  // Global position
    uint32_t layer_count;                // How many layers contributed
    uint64_t last_updated;               // Timestamp or step number
    
    attention_score_data() 
        : aggregated_score(0.0f), 
          position(0), 
          layer_count(0), 
          last_updated(0) {}
};

// Structure for reverse attention trim parameters (internal)
struct reverse_attention_trim_params {
    float trim_threshold;                // Percentage to trim (0.0-1.0)
    float min_attention_score;           // Minimum attention score to keep
    float recent_token_weight;           // Weight multiplier for recent tokens
    float system_prompt_weight;          // Weight for system prompt tokens
    int min_tokens_to_keep;              // Minimum tokens to keep
    bool aggregate_across_layers;        // Average scores across layers
    bool use_cumulative_score;           // Use cumulative or last score
    bool preserve_system_prompt;         // Never trim system prompt
    int system_prompt_end_pos;           // End position of system prompt
    
    reverse_attention_trim_params()
        : trim_threshold(0.25f),
          min_attention_score(0.01f),
          recent_token_weight(1.5f),
          system_prompt_weight(2.0f),
          min_tokens_to_keep(100),
          aggregate_across_layers(true),
          use_cumulative_score(true),
          preserve_system_prompt(true),
          system_prompt_end_pos(0) {}
};

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    uint32_t get_non_empty_cell_count() const override;
    void debug_cell_states() const override;

    static uint32_t get_padding(const llama_cparams & cparams);
    bool move_cell(uint32_t stream_id, uint32_t src_idx, uint32_t dst_idx);
    void update_external_position_tracking(const std::map<llama_pos, uint32_t>& position_remapping);
    void debug_attention_pattern() const;
    void renumber_global_positions();

    llama_pos get_current_max_position() const override;
    uint32_t get_current_used_cells() const;
    void update_internal_counters();
    void update_sequence_position_tracking();
    llama_pos get_api_max_position() const;
    
    // Reverse attention trimming methods
    void trim_reverse_attention(
        int trim_percentage,
        const reverse_attention_trim_params& params,
        const std::map<llama_pos, std::string>* token_mapping = nullptr);
    
    void trim_reverse_attention_ex(
        const llama_reverse_attention_params* api_params,
        const std::map<llama_pos, std::string>* token_mapping = nullptr);
    
    // Более простая версия с минимальными параметрами
    void trim_reverse_attention_simple(int trim_percentage) override;
    
    // Attention tracking methods
    void register_attention_scores(
        int layer,
        const std::vector<float>& attention_matrix,
        size_t n_kv,
        size_t n_tokens) override;
    
    void clear_attention_scores() override;
    
    // Get attention statistics
    attention_statistics get_attention_statistics() const;
    
    // Set attention callback
    void set_attention_callback(
        llama_attention_callback callback,
        void* user_data) override;
    
    // Enable/disable attention tracking
    void enable_attention_tracking(bool enabled) override;
    
    // Check if attention tracking is enabled
    bool is_attention_tracking_enabled() const override;

    // Вспомогательные методы для расчета важности
    float calculate_token_importance(
        llama_pos position,
        const attention_score_data& scores,
        const reverse_attention_trim_params& params) const;

    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    llama_kv_cache(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache() = default;

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
    
    // Override base class methods
    uint32_t get_size() const override;
    uint32_t get_n_seq_max() const override { return n_seq_max; }
    
    // Access to internal components
    llama_kv_cache * get_kv_cache() const override { return const_cast<llama_kv_cache*>(this); }
    
    // Stats and debug
    uint32_t get_used_cells() const override { return get_current_used_cells(); }
    
    // Reverse attention and trimming
    void trim_random(int trim_percentage, const std::map<llama_pos, std::string>* token_mapping = nullptr) override;
    bool compact() override;
    
    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    std::vector<llama_kv_cells> v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    bool is_masked_swa(llama_pos p0, llama_pos p1) const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count);
    
    // Helper methods for reverse attention
    std::vector<llama_pos> select_tokens_to_trim(
        const std::map<llama_pos, float>& importance_scores,
        const reverse_attention_trim_params& params) const;
    
    void update_attention_statistics();

    // Вспомогательные методы для reverse attention
    std::vector<llama_pos> select_tokens_to_trim_reverse(
        const reverse_attention_trim_params& params,
        int target_evictions) const;
    
    void evict_tokens_by_importance(
        const std::vector<llama_pos>& tokens_to_evict,
        const std::map<llama_pos, std::string>* token_mapping);

    // Reverse Attention Data Members
    std::map<llama_pos, attention_score_data> attention_scores_;
    mutable std::mutex attention_scores_mutex_;
    
    llama_attention_callback attention_callback_ = nullptr;
    void* attention_callback_user_data_ = nullptr;
    bool attention_tracking_enabled_ = false;
    
    attention_statistics current_stats_;
    mutable std::mutex stats_mutex_;
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch procesing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;
    const llama_context* get_context() const override { return lctx; }
    
    // Common methods from base interface
    uint32_t get_head() const override { return 0; } // KV cache не использует head так же как recurrent
    uint32_t get_size() const override { return kv ? kv->get_size() : 0; }
    uint32_t get_n_kv() const override;
    uint32_t get_n_rs() const override { return 0; }
    int32_t get_rs_z() const override { return -1; }
    uint32_t get_used_cells() const override { return kv ? kv->get_used_cells() : 0; }
    uint32_t get_total_cells() const override { return kv ? kv->get_size() : 0; }
    llama_pos get_max_position() const override { return kv ? kv->get_current_max_position() : 0; }
    
    // Tensor access
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_r_l(int32_t il) const override { return nullptr; }
    ggml_tensor * get_s_l(int32_t il) const override { return nullptr; }
    
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
    const llama_kv_cache_context * as_kv_cache() const override { return this; }
    const llama_kv_cache_iswa_context * as_kv_cache_iswa() const override { return nullptr; }
    const llama_memory_recurrent_context * as_recurrent() const override { return nullptr; }
    const llama_memory_hybrid_context * as_hybrid() const override { return nullptr; }
    
    // Helper method
    int32_t s_copy(int i) const override { return -1; }

    //
    // llama_kv_cache_context specific API
    //

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
