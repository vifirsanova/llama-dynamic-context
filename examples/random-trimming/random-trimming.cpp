#include "llama.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <algorithm>
#include <cctype>

// Global token tracking
std::map<llama_pos, std::string> position_to_token;
std::map<llama_pos, llama_token> position_to_token_id;
std::stringstream generation_output;
std::vector<llama_pos> last_trimmed_positions;
std::string last_trim_reason;

// Function declarations
void debug_kv_cache_state(llama_context* ctx, const char* phase);
void print_real_attention_visualization();
//void print_matrix_shape_analysis(llama_context* ctx);
void print_trimming_impact_analysis();
void parse_trimming_output(const std::string& stderr_content);
void print_deleted_tokens_report();
void cleanup_token_mapping(llama_context* ctx);
static void print_usage(int argc, char ** argv);

void debug_kv_cache_state(llama_context* ctx, const char* phase) {
    fprintf(stderr, "\n[KV_CACHE_STATE %s]\n", phase);
    fprintf(stderr, "Context pointer: %p\n", (void*)ctx);

    int total_capacity = llama_n_ctx(ctx);

    // Use the available API - this should match what renumber_global_positions() finds
    auto* memory = llama_get_memory(ctx);
    int max_seq_pos = llama_memory_seq_pos_max(memory, 0);

    // For cells_used, we'll use max_seq_pos + 1 as an estimate
    // This assumes dense packing (which compaction should achieve)
    int cells_used = max_seq_pos + 1;

    fprintf(stderr, "Total capacity: %d tokens\n", total_capacity);
    //fprintf(stderr, "Max sequence position: %d\n", max_seq_pos);
    fprintf(stderr, "Cells used: %d/%d\n", cells_used, total_capacity);
    fprintf(stderr, "Utilization: %.1f%%\n", (float)cells_used / total_capacity * 100.0f);

    // Check if compaction worked - if positions are dense, this should be true
    if (max_seq_pos + 1 == cells_used) {
        fprintf(stderr, "Cache appears to be dense (no gaps)\n");
    } else {
        fprintf(stderr, "Cache may have gaps\n");
    }

    // ADD DEBUG: Check if the reported max position makes sense
    //fprintf(stderr, "[DEBUG] llama_memory_seq_pos_max returned: %d\n", max_seq_pos);
}

void parse_trimming_output(const std::string& stderr_content) {
    last_trimmed_positions.clear();
    //fprintf(stderr, "DEBUG: parse_trimming_output called, stderr_content length: %zu\n", stderr_content.length());

    //if (stderr_content.length() > 0) {
    //    fprintf(stderr, "DEBUG: First 500 chars of stderr:\n%.500s\n", stderr_content.c_str());
    //}

    size_t pos_start = stderr_content.find("[TRIMMING_POSITIONS]");
    if (pos_start != std::string::npos) {
        //fprintf(stderr, "DEBUG: Found [TRIMMING_POSITIONS] at position %zu\n", pos_start);
        size_t pos_end = stderr_content.find("\n", pos_start);
        if (pos_end != std::string::npos) {
            std::string pos_line = stderr_content.substr(pos_start, pos_end - pos_start);
            //fprintf(stderr, "DEBUG: Position line: %s\n", pos_line.c_str());

            size_t start = pos_line.find("]");
            if (start != std::string::npos) {
                std::string positions_str = pos_line.substr(start + 1);
                //fprintf(stderr, "DEBUG: Positions string: '%s'\n", positions_str.c_str());
                std::istringstream iss(positions_str);
                llama_pos pos;
                while (iss >> pos) {
                    last_trimmed_positions.push_back(pos);
                    //fprintf(stderr, "[DEBUG] Parsed position: %d\n", pos);
                }
            }
        }
      }
//    } else {
//        fprintf(stderr, "DEBUG: [TRIMMING_POSITIONS] not found in stderr content\n");
//    }

//    fprintf(stderr, "[DEBUG] Parsed %zu trimmed positions\n", last_trimmed_positions.size());
}

void print_deleted_tokens_report() {
    fprintf(stderr, "\n[DELETED_TOKENS_REPORT]");

    if (last_trimmed_positions.empty()) {
        fprintf(stderr, "No tokens were trimmed in the last operation\n");
        return;
    }

    //fprintf(stderr, "\n=== DELETED TOKENS REPORT (%s) ===\n", last_trim_reason.c_str());
    fprintf(stderr, "Trimmed %zu tokens\n", last_trimmed_positions.size());

    for (llama_pos pos : last_trimmed_positions) {
        auto token_it = position_to_token.find(pos);
        auto token_id_it = position_to_token_id.find(pos);

        if (token_it != position_to_token.end() && token_id_it != position_to_token_id.end()) {
            fprintf(stderr, "Position %d: '%s' (ID: %d)\n",
                   pos, token_it->second.c_str(), token_id_it->second);
        } else {
            fprintf(stderr, "Position %d: [UNKNOWN - already deleted or not tracked]\n", pos);
        }
    }
    //fprintf(stderr, "=== END REPORT ===\n\n");
}

void cleanup_token_mapping(llama_context* ctx) {
    int current_size = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;

    auto it = position_to_token.begin();
    while (it != position_to_token.end()) {
        if (it->first >= current_size) {
            it = position_to_token.erase(it);
        } else {
            ++it;
        }
    }

    auto it2 = position_to_token_id.begin();
    while (it2 != position_to_token_id.end()) {
        if (it2->first >= current_size) {
            it2 = position_to_token_id.erase(it2);
        } else {
            ++it2;
        }
    }
}

static void print_usage(int, char ** argv) {
    fprintf(stderr, "\nexample usage:\n");
    fprintf(stderr, "\n    %s -m model.gguf [-c context_size] [-ngl n_gpu_layers] [-trim_pct percentage]\n", argv[0]);
    fprintf(stderr, "\n    trim_pct: percentage of cache to trim (default: 25)\n");
    fprintf(stderr, "\n");
}

void print_real_attention_visualization() {
    if (position_to_token.empty()) {
        fprintf(stderr, "[ATTENTION_VISUAL] No tokens to visualize\n");
        return;
    }

    fprintf(stderr, "\n=== CACHE VISUALIZATION ===\n");
    llama_pos max_pos = 0;
    for (const auto& entry : position_to_token) {
        if (entry.first > max_pos) max_pos = entry.first;
    }

    int display_range = std::min(60, (int)max_pos + 1);
    fprintf(stderr, "CURRENT POSITIONS (first %d):\n", display_range);

    fprintf(stderr, "Pos: ");
    for (int pos = 0; pos < display_range; pos++) {
        if (pos % 10 == 0) {
            fprintf(stderr, "|%-2d", pos);
        } else {
            fprintf(stderr, " %-2d", pos);
        }
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "Tokens:");
    for (int pos = 0; pos < display_range; pos++) {
        if (position_to_token.find(pos) != position_to_token.end()) {
            fprintf(stderr, " ██");
        } else {
            fprintf(stderr, " __");
        }
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "Trimmed:");
    for (int pos = 0; pos < display_range; pos++) {
        bool was_trimmed = std::find(last_trimmed_positions.begin(),
                                   last_trimmed_positions.end(), pos) != last_trimmed_positions.end();
        if (was_trimmed) {
            fprintf(stderr, " X ");
        } else if (position_to_token.find(pos) != position_to_token.end()) {
            fprintf(stderr, "   ");
        } else {
            fprintf(stderr, " . ");
        }
    }
    fprintf(stderr, "\n");

    int total_positions = max_pos + 1;
    int occupied_positions = position_to_token.size();
    int gaps = total_positions - occupied_positions;

    fprintf(stderr, "\nSTATISTICS:\n");
    fprintf(stderr, " - Total position range: 0-%d (%d positions)\n", max_pos, total_positions);
    fprintf(stderr, " - Occupied positions: %d (%.1f%%)\n", occupied_positions,
            (float)occupied_positions / total_positions * 100);
    fprintf(stderr, " - Gaps: %d (%.1f%%)\n", gaps, (float)gaps / total_positions * 100);
    fprintf(stderr, " - Recently trimmed: %zu tokens\n", last_trimmed_positions.size());
}

/*
void print_matrix_shape_analysis(llama_context* ctx) {
    fprintf(stderr, "\n=== K,V,Q MATRIX SHAPE ANALYSIS ===\n");
    int n_ctx = llama_n_ctx(ctx);
    int n_layers = 32;
    int n_heads = 32;
    int d_head = 128;

    fprintf(stderr, "NOTE: Matrix shapes are ESTIMATES based on typical architecture\n");
    fprintf(stderr, "      Actual shapes depend on specific model: %s\n", model_path.c_str());
    fprintf(stderr, "      Layers: %d (reported by model)\n", n_layers);

    fprintf(stderr, "TYPICAL MATRIX SHAPES (for context size %d):\n", n_ctx);
    fprintf(stderr, "K matrix per layer: [%d, %d] = %d elements\n",
            n_heads * d_head, n_ctx, n_heads * d_head * n_ctx);
    fprintf(stderr, "V matrix per layer: [%d, %d] = %d elements\n",
            n_heads * d_head, n_ctx, n_heads * d_head * n_ctx);
    fprintf(stderr, "Q matrix per step:  [%d, 1]   = %d elements\n",
            n_heads * d_head, n_heads * d_head);

    fprintf(stderr, "\nTOTAL STORAGE (all %d layers):\n", n_layers);
    size_t kv_per_layer = n_heads * d_head * n_ctx * sizeof(float);
    size_t total_kv = kv_per_layer * n_layers * 2;
    fprintf(stderr, "K,V cache: ~%.1f MB per layer, %.1f MB total\n",
            kv_per_layer / (1024.0 * 1024.0), total_kv / (1024.0 * 1024.0));

    fprintf(stderr, "\nIMPACT OF TRIMMING:\n");
    fprintf(stderr, " - K,V matrices have fixed shape [embedding, %d]\n", n_ctx);
    fprintf(stderr, " - Trimmed positions = unused columns in K,V\n");
    fprintf(stderr, " - Wasted computation on -inf attention scores\n");
    fprintf(stderr, " - Q vectors search through sparse context\n");
}
*/

void print_trimming_impact_analysis() {
    //fprintf(stderr, "\n=== DEBUG: ENTERING TRIMMING IMPACT ANALYSIS ===\n");
    //fprintf(stderr, "last_trimmed_positions size: %zu\n", last_trimmed_positions.size());

    if (last_trimmed_positions.empty()) {
        //fprintf(stderr, "DEBUG: No trimmed positions to analyze\n");
        return;
    }

    //fprintf(stderr, "DEBUG: First 5 trimmed positions: ");
    //for (int i = 0; i < std::min(5, (int)last_trimmed_positions.size()); i++) {
    //    fprintf(stderr, "%d ", last_trimmed_positions[i]);
    //}
    //fprintf(stderr, "\n");

    std::vector<llama_pos> sorted_trimmed = last_trimmed_positions;
    std::sort(sorted_trimmed.begin(), sorted_trimmed.end());

    llama_pos max_pos = 0;
    for (const auto& entry : position_to_token) {
        if (entry.first > max_pos) max_pos = entry.first;
    }
    //fprintf(stderr, "DEBUG: Max position in tracking: %d\n", max_pos);

    if (max_pos > 0) {
        int early_cutoff = max_pos / 3;
        int late_cutoff = 2 * max_pos / 3;

        int early_trimmed = 0, middle_trimmed = 0, late_trimmed = 0;

        for (llama_pos pos : sorted_trimmed) {
            if (pos < early_cutoff) early_trimmed++;
            else if (pos < late_cutoff) middle_trimmed++;
            else late_trimmed++;
        }

        fprintf(stderr, "TRIMMED TOKEN DISTRIBUTION:\n");
        fprintf(stderr, " - Early (0-%d): %d tokens (%.1f%%)\n",
                early_cutoff, early_trimmed, (float)early_trimmed/sorted_trimmed.size()*100);
        fprintf(stderr, " - Middle (%d-%d): %d tokens (%.1f%%)\n",
                early_cutoff, late_cutoff, middle_trimmed, (float)middle_trimmed/sorted_trimmed.size()*100);
        fprintf(stderr, " - Late (%d-%d): %d tokens (%.1f%%)\n",
                late_cutoff, max_pos, late_trimmed, (float)late_trimmed/sorted_trimmed.size()*100);
    }

    std::map<std::string, int> token_categories;
    int found_tokens = 0, missing_tokens = 0;

    for (llama_pos pos : sorted_trimmed) {
        auto it = position_to_token.find(pos);
        if (it != position_to_token.end()) {
            found_tokens++;
            std::string token = it->second;
            //fprintf(stderr, "DEBUG: Analyzing token at pos %d: '%s'\n", pos, token.c_str());

            if (token.length() == 1 && ispunct(token[0])) {
                token_categories["punctuation"]++;
            } else if (token.find(' ') != std::string::npos) {
                token_categories["multi-word"]++;
            } else if (!token.empty() && isupper(token[0])) {
                token_categories["capitalized"]++;
            } else if (token.length() <= 2) {
                token_categories["short"]++;
            } else {
                token_categories["other"]++;
            }
        } else {
            missing_tokens++;
            fprintf(stderr, "DEBUG: Missing token at pos %d\n", pos);
        }
    }

    //fprintf(stderr, "DEBUG: Found tokens: %d, Missing tokens: %d\n", found_tokens, missing_tokens);

    fprintf(stderr, "\nTRIMMED TOKEN CATEGORIES:\n");
    for (const auto& cat : token_categories) {
        fprintf(stderr, " - %s: %d (%.1f%%)\n",
                cat.first.c_str(), cat.second,
                (float)cat.second/sorted_trimmed.size()*100);
    }

    //fprintf(stderr, "=== DEBUG: END OF TRIMMING IMPACT ANALYSIS ===\n\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    int ngl = 99;
    int n_ctx = 2048;
    int trim_pct = 25;

    // parse command line arguments
    for (int i = 1; i < argc; i++) {
        try {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-c") == 0) {
                if (i + 1 < argc) {
                    n_ctx = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    ngl = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-trim_pct") == 0) {
                if (i + 1 < argc) {
                    trim_pct = std::stoi(argv[++i]);
                    if (trim_pct < 1 || trim_pct > 99) {
                        fprintf(stderr, "error: trim percentage must be between 1 and 99\n");
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } catch (std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            print_usage(argc, argv);
            return 1;
        }
    }
    if (model_path.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_layers = llama_model_n_layer(model);
    fprintf(stderr, "Model architecture: n_layers=%d\n", n_layers);

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Helper function as lambda
    auto get_actual_generation_start_pos = [](llama_context* ctx) -> llama_pos {
        auto* memory = llama_get_memory(ctx);
        llama_pos max_pos = llama_memory_seq_pos_max(memory, 0);

        if (max_pos == -1) {
            return 0;
        }

        return max_pos + 1;
    };

    // Enhanced trim function using temporary file
    auto apply_trimming = [&](const char* reason) {
        fprintf(stderr, "[TRIMMING_PREDICTION] Planning to free ~%d positions (%d%% of %d)\n",
                n_ctx * trim_pct / 100, trim_pct, n_ctx);
        if (trim_pct == 0) return;

        last_trim_reason = reason;

        //fprintf(stderr, "\n=== Applying KV cache trimming (%s) ===\n", reason);

        //int cells_before = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        //fprintf(stderr, "Current cells: %d\n", cells_before);

        //debug_kv_cache_state(ctx, "BEFORE_TRIM");
        print_real_attention_visualization();

        // Use temporary file instead of fork
        std::string temp_file = "/tmp/llama_trim_output_" + std::to_string(getpid());

        // Redirect stderr to file, trim, then restore
        fflush(stderr);
        int saved_stderr = dup(STDERR_FILENO);
        FILE* temp_output = fopen(temp_file.c_str(), "w");
        dup2(fileno(temp_output), STDERR_FILENO);

        // Perform trimming in main process
        llama_kv_cache_trim_random(ctx, trim_pct);

        // Restore stderr
        fflush(stderr);
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);
        fclose(temp_output);

        // Read and parse the output
        std::ifstream file(temp_file);
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();
        std::remove(temp_file.c_str());

        std::string trim_output = buffer.str();
        //fprintf(stderr, "%s", trim_output.c_str());

        parse_trimming_output(trim_output);

        //fprintf(stderr, "=== Trimming completed ===\n\n");
        print_deleted_tokens_report();

        print_trimming_impact_analysis();

        for (llama_pos pos : last_trimmed_positions) {
            position_to_token.erase(pos);
            position_to_token_id.erase(pos);
        }

        llama_kv_cache_compact(ctx);
        debug_kv_cache_state(ctx, "AFTER_TRIM");
    };

    // helper function to evaluate a prompt and generate a response
    auto generate = [&](const std::string & prompt) {
        std::string response;
        std::vector<llama_token> all_tokens;

        //fprintf(stderr, "[GENERATION START]\n");
        debug_kv_cache_state(ctx, "GENERATION_START");

        // Get the ACTUAL current max position from KV cache
        llama_pos start_pos = get_actual_generation_start_pos(ctx);
        const bool is_first = (start_pos == 0);

        fprintf(stderr, "[POSITION_TRACKING] Starting at position %d (%s generation)\n", start_pos, is_first ? "first" : "continued");

        // tokenize the prompt
        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            GGML_ABORT("failed to tokenize the prompt\n");
        }

        // Track prompt tokens
        fprintf(stderr, "[TOKEN_TRACKING] Tracking %zu prompt tokens starting at position %d\n", prompt_tokens.size(), start_pos);

        fprintf(stderr, "[DEBUG] KV cache max position before generation: %d\n", llama_memory_seq_pos_max(llama_get_memory(ctx), 0));
        //fprintf(stderr, "[DEBUG] Expected first token position: %d\n", start_pos);

        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            llama_pos pos = start_pos + i;
            char buf[256];
            int n = llama_token_to_piece(vocab, prompt_tokens[i], buf, sizeof(buf), 0, true);
            if (n > 0) {
                std::string piece(buf, n);
                position_to_token[pos] = piece;
                position_to_token_id[pos] = prompt_tokens[i];
                //fprintf(stderr, "[TOKEN_TRACK] Prompt token at pos %d: '%s' (id: %d)\n",
                //       pos, piece.c_str(), prompt_tokens[i]);
            }
        }

        all_tokens.insert(all_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());

        // prepare a batch for the prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;

        int generation_step = 0;

        // NEW: Clear generation output for this turn
        generation_output.str("");
        generation_output.clear();

        while (true) {
            int ret = llama_decode(ctx, batch);
            if (ret != 0) {
                GGML_ABORT("failed to decode, ret = %d\n", ret);
            }

            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                GGML_ABORT("failed to convert token to piece\n");
            }
            std::string piece(buf, n);

            // NEW: Dual output - both to stdout and to buffer
            printf("%s", piece.c_str());
            fflush(stdout);
            generation_output << piece;
            response += piece;

            // Track generated token position
            llama_pos pos = start_pos + prompt_tokens.size() + generation_step;

            if (generation_step == 0) {
                fprintf(stderr, "[DEBUG] First generated token at position: %d\n[CONTINUE GENERATION]\n", pos);
            }

            position_to_token[pos] = piece;
            position_to_token_id[pos] = new_token_id;
            //fprintf(stderr, "[TOKEN_TRACK] Generated token at pos %d: '%s' (id: %d)\n",
            //       pos, piece.c_str(), new_token_id);

            all_tokens.push_back(new_token_id);
            generation_step++;
            batch = llama_batch_get_one(&new_token_id, 1);
        }

        // DEBUG: Verify final state
        llama_pos final_max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        fprintf(stderr, "\n[DEBUG] KV cache max position after generation: %d\n", final_max_pos);
        //fprintf(stderr, "[DEBUG] Expected final position: %zu\n", start_pos + all_tokens.size() - 1);

        fprintf(stderr, "[GENERATION END] Generated %zu tokens\n", all_tokens.size() - prompt_tokens.size());
        debug_kv_cache_state(ctx, "GENERATION_END");

        return response;
    };

    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(n_ctx);
    int prev_len = 0;

    // Print initial configuration
    fprintf(stderr, "\n=== Configuration ===\n");
    fprintf(stderr, "Model: %s\n", model_path.c_str());
    fprintf(stderr, "Context size: %d positions\n", n_ctx);
    fprintf(stderr, "GPU layers: %d\n", ngl);
    fprintf(stderr, "Trim percentage: %d%%\n", trim_pct);
    fprintf(stderr, "Model layers: %d\n", n_layers);
    debug_kv_cache_state(ctx, "INITIAL");

    while (true) {
        // get user input
        printf("\033[32m> \033[0m");
        std::string user;
        std::getline(std::cin, user);

        if (user.empty()) {
            break;
        }

        const char * tmpl = llama_model_chat_template(model, nullptr);

        // add the user input to the message list and format it
        messages.push_back({"user", strdup(user.c_str())});
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        if (new_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            return 1;
        }

        // remove previous messages to obtain the prompt to generate the response
        std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

        // generate a response
        printf("\033[33m");
        std::string response = generate(prompt);
        printf("\n\033[0m");

        // NEW: You can now access the generated text from generation_output.str()
        //fprintf(stderr, "[GENERATED_TEXT] %s\n", generation_output.str().c_str());

        // ALWAYS TRIM AFTER GENERATION
        //fprintf(stderr, "[AUTO-TRIM] Applying trim after generation\n");
        apply_trimming("after_generation");

        // add the response to the messages
        messages.push_back({"assistant", strdup(response.c_str())});
        prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
        if (prev_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            return 1;
        }
    }

    // Print final state
    fprintf(stderr, "\n=== Final State ===\n");
    debug_kv_cache_state(ctx, "FINAL");

    // Clear token tracking
    position_to_token.clear();
    position_to_token_id.clear();

    // free resources
    for (auto & msg : messages) {
        free(const_cast<char *>(msg.content));
    }
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
