#include "llama-memory-hybrid.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
        std::unique_ptr<llama_memory_i> attn,
        std::unique_ptr<llama_memory_i> recr) :
    attn(std::move(attn)),
    recr(std::move(recr)) {}

llama_memory_hybrid::~llama_memory_hybrid() = default;

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // TODO: non-sequential equal split can be done if using unified KV cache
                //       for simplicity, we always use sequential equal split for now
                ubatch = balloc.split_equal(n_ubatch, true);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // Создаем контексты для обработки батча
        auto ctx_attn = attn->init_batch(balloc, n_ubatch, embd_all);
        auto ctx_recr = recr->init_batch(balloc, n_ubatch, embd_all);

        // Проверяем успешность создания контекстов
        if (ctx_recr->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: failed to create recurrent batch context\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        if (ctx_attn->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: failed to create attention batch context\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_context>(
                std::move(ctx_attn),
                std::move(ctx_recr),
                std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    auto ctx = std::make_unique<llama_memory_hybrid_context>(this);
    
    if (ctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS) {
        // Apply updates to both components
        auto ctx_attn_update = attn->init_update(lctx, optimize);
        auto ctx_recr_update = recr->init_update(lctx, optimize);
        
        if (ctx_attn_update && ctx_recr_update) {
            ctx_attn_update->apply();
            ctx_recr_update->apply();
        }
    }
    
    return ctx;
}

bool llama_memory_hybrid::get_can_shift() const {
    return attn->get_can_shift() && recr->get_can_shift();
}

uint32_t llama_memory_hybrid::get_size() const {
    return attn->get_size() + recr->get_size();
}

uint32_t llama_memory_hybrid::get_n_seq_max() const {
    return std::max(attn->get_n_seq_max(), recr->get_n_seq_max());
}

llama_memory_i * llama_memory_hybrid::get_attn() const {
    return attn.get();
}

llama_memory_i * llama_memory_hybrid::get_recr() const {
    return recr.get();
}

llama_kv_cache * llama_memory_hybrid::get_kv_cache() const {
    return dynamic_cast<llama_kv_cache*>(attn.get());
}

llama_memory_recurrent * llama_memory_hybrid::get_recurrent() const {
    return dynamic_cast<llama_memory_recurrent*>(recr.get());
}

void llama_memory_hybrid::clear(bool data) {
    attn->clear(data);
    recr->clear(data);
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    attn->seq_keep(seq_id);
    recr->seq_keep(seq_id);
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    attn->seq_add(seq_id, p0, p1, shift);
    recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    attn->seq_div(seq_id, p0, p1, d);
    recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(attn->seq_pos_min(seq_id), recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(attn->seq_pos_max(seq_id), recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = attn->memory_breakdown();
    for (const auto & buft_size : recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

uint32_t llama_memory_hybrid::get_used_cells() const {
    return attn->get_used_cells() + recr->get_used_cells();
}

uint32_t llama_memory_hybrid::get_non_empty_cell_count() const {
    return attn->get_non_empty_cell_count() + recr->get_non_empty_cell_count();
}

void llama_memory_hybrid::debug_cell_states() const {
    LLAMA_LOG_INFO("=== Attention Cache ===\n");
    attn->debug_cell_states();
    LLAMA_LOG_INFO("=== Recurrent Cache ===\n");
    recr->debug_cell_states();
}

llama_pos llama_memory_hybrid::get_current_max_position() const {
    return std::max(attn->get_current_max_position(), recr->get_current_max_position());
}

void llama_memory_hybrid::trim_random(int trim_percentage, const std::map<llama_pos, std::string>* token_mapping) {
    attn->trim_random(trim_percentage, token_mapping);
    recr->trim_random(trim_percentage, token_mapping);
}

void llama_memory_hybrid::trim_reverse_attention_simple(int trim_percentage) {
    attn->trim_reverse_attention_simple(trim_percentage);
    recr->trim_reverse_attention_simple(trim_percentage);
}

bool llama_memory_hybrid::compact() {
    bool attn_compacted = attn->compact();
    bool recr_compacted = recr->compact();
    return attn_compacted || recr_compacted;
}

void llama_memory_hybrid::enable_attention_tracking(bool enabled) {
    attn->enable_attention_tracking(enabled);
    recr->enable_attention_tracking(enabled);
}

bool llama_memory_hybrid::is_attention_tracking_enabled() const {
    return attn->is_attention_tracking_enabled() || recr->is_attention_tracking_enabled();
}

void llama_memory_hybrid::set_attention_callback(
    llama_attention_callback callback,
    void* user_data) {
    attn->set_attention_callback(callback, user_data);
    recr->set_attention_callback(callback, user_data);
}

void llama_memory_hybrid::clear_attention_scores() {
    attn->clear_attention_scores();
    recr->clear_attention_scores();
}

void llama_memory_hybrid::register_attention_scores(
    int layer,
    const std::vector<float>& attention_matrix,
    size_t n_kv,
    size_t n_tokens) {
    attn->register_attention_scores(layer, attention_matrix, n_kv, n_tokens);
    recr->register_attention_scores(layer, attention_matrix, n_kv, n_tokens);
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        attn->state_write(io, seq_id, flags);
    }
    recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        attn->state_read(io, seq_id, flags);
    }
    recr->state_read(io, seq_id, flags);
}

// ============================================================
// llama_memory_hybrid_context
// ============================================================

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : 
    status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    ctx_attn(mem->get_attn()->init_full()),
    ctx_recr(mem->get_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
            llama_memory_context_ptr ctx_attn,
            llama_memory_context_ptr ctx_recr,
            std::vector<llama_ubatch> ubatches) :
    ubatches(std::move(ubatches)),
    ctx_attn(std::move(ctx_attn)),
    ctx_recr(std::move(ctx_recr)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::~llama_memory_hybrid_context() = default;

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    
    if (!ctx_attn || !ctx_recr) {
        return false;
    }

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    if (ctx_attn) {
        res = res && ctx_attn->apply();
    }
    
    if (ctx_recr) {
        res = res && ctx_recr->apply();
    }

    return res;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_context* llama_memory_hybrid_context::get_context() const {
    // Try to get context from attention component first
    if (ctx_attn) {
        const llama_context* ctx = ctx_attn->get_context();
        if (ctx) {
            return ctx;
        }
    }
    
    // Fall back to recurrent component
    if (ctx_recr) {
        return ctx_recr->get_context();
    }
    
    return nullptr;
}

uint32_t llama_memory_hybrid_context::get_head() const {
    if (ctx_recr) {
        return ctx_recr->get_head();
    }
    return 0;
}

uint32_t llama_memory_hybrid_context::get_size() const {
    uint32_t size = 0;
    if (ctx_attn) size += ctx_attn->get_total_cells();
    if (ctx_recr) size += ctx_recr->get_total_cells();
    return size;
}

uint32_t llama_memory_hybrid_context::get_n_kv() const {
    if (ctx_attn) {
        return ctx_attn->get_n_kv();
    }
    return 0;
}

uint32_t llama_memory_hybrid_context::get_n_rs() const {
    if (ctx_recr) {
        return ctx_recr->get_n_rs();
    }
    return 0;
}

int32_t llama_memory_hybrid_context::get_rs_z() const {
    if (ctx_recr) {
        return ctx_recr->get_rs_z();
    }
    return -1;
}

uint32_t llama_memory_hybrid_context::get_used_cells() const {
    uint32_t used = 0;
    if (ctx_attn) used += ctx_attn->get_used_cells();
    if (ctx_recr) used += ctx_recr->get_used_cells();
    return used;
}

uint32_t llama_memory_hybrid_context::get_total_cells() const {
    return get_size();
}

llama_pos llama_memory_hybrid_context::get_max_position() const {
    llama_pos max_pos = 0;
    if (ctx_attn) {
        llama_pos attn_pos = ctx_attn->get_max_position();
        if (attn_pos > max_pos) max_pos = attn_pos;
    }
    if (ctx_recr) {
        llama_pos recr_pos = ctx_recr->get_max_position();
        if (recr_pos > max_pos) max_pos = recr_pos;
    }
    return max_pos;
}

ggml_tensor * llama_memory_hybrid_context::get_k(ggml_context * ctx, int32_t il) const {
    if (ctx_attn) {
        return ctx_attn->get_k(ctx, il);
    }
    return nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_v(ggml_context * ctx, int32_t il) const {
    if (ctx_attn) {
        return ctx_attn->get_v(ctx, il);
    }
    return nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_r_l(int32_t il) const {
    if (ctx_recr) {
        return ctx_recr->get_r_l(il);
    }
    return nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_s_l(int32_t il) const {
    if (ctx_recr) {
        return ctx_recr->get_s_l(il);
    }
    return nullptr;
}

ggml_tensor * llama_memory_hybrid_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    if (ctx_attn) {
        return ctx_attn->cpy_k(ctx, k_cur, k_idxs, il);
    }
    return nullptr;
}

ggml_tensor * llama_memory_hybrid_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    if (ctx_attn) {
        return ctx_attn->cpy_v(ctx, v_cur, v_idxs, il);
    }
    return nullptr;
}

ggml_tensor * llama_memory_hybrid_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    if (ctx_attn) {
        return ctx_attn->build_input_k_idxs(ctx, ubatch);
    }
    return nullptr;
}

ggml_tensor * llama_memory_hybrid_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    if (ctx_attn) {
        return ctx_attn->build_input_v_idxs(ctx, ubatch);
    }
    return nullptr;
}

void llama_memory_hybrid_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (ctx_attn) {
        ctx_attn->set_input_k_idxs(dst, ubatch);
    }
}

void llama_memory_hybrid_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (ctx_attn) {
        ctx_attn->set_input_v_idxs(dst, ubatch);
    }
}

void llama_memory_hybrid_context::set_input_k_shift(ggml_tensor * dst) const {
    if (ctx_attn) {
        ctx_attn->set_input_k_shift(dst);
    }
}

void llama_memory_hybrid_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    if (ctx_attn) {
        ctx_attn->set_input_kq_mask(dst, ubatch, causal_attn);
    }
}

void llama_memory_hybrid_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (ctx_attn) {
        ctx_attn->set_input_pos_bucket(dst, ubatch);
    }
}

const llama_kv_cache_context * llama_memory_hybrid_context::as_kv_cache() const {
    if (ctx_attn) {
        return ctx_attn->as_kv_cache();
    }
    return nullptr;
}

const llama_kv_cache_iswa_context * llama_memory_hybrid_context::as_kv_cache_iswa() const {
    if (ctx_attn) {
        return ctx_attn->as_kv_cache_iswa();
    }
    return nullptr;
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::as_recurrent() const {
    if (ctx_recr) {
        return ctx_recr->as_recurrent();
    }
    return nullptr;
}

const llama_memory_hybrid_context * llama_memory_hybrid_context::as_hybrid() const {
    return this;
}

int32_t llama_memory_hybrid_context::s_copy(int i) const {
    if (ctx_recr) {
        return ctx_recr->s_copy(i);
    }
    return -1;
}

const llama_memory_context_i * llama_memory_hybrid_context::get_attn() const {
    return ctx_attn.get();
}

const llama_memory_context_i * llama_memory_hybrid_context::get_recr() const {
    return ctx_recr.get();
}
