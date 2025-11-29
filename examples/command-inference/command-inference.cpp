#include "llama.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <fcntl.h>

int main(int argc, char ** argv) {
    int dev_null = open("/dev/null", O_WRONLY);
    if (dev_null != -1) {
        dup2(dev_null, STDERR_FILENO);
        close(dev_null);
    }
    std::string model_path;
    int ngl = 99;
    int n_ctx = 2048;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                return 1;
            }
        } else if (strcmp(argv[i], "-c") == 0) {
            if (i + 1 < argc) {
                n_ctx = std::stoi(argv[++i]);
            } else {
                return 1;
            }
        } else if (strcmp(argv[i], "-ngl") == 0) {
            if (i + 1 < argc) {
                ngl = std::stoi(argv[++i]);
            } else {
                return 1;
            }
        }
    }
    if (model_path.empty()) {
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        return 1;
    }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    while (true) {
        printf("> ");
        std::string user;
        std::getline(std::cin, user);

        if (user.empty()) {
            break;
        }

        // Few-shot промпт с примерами
        std::string full_prompt = 
            "Convert English to Linux command:\n"
            "Input: show files in directory\n"
            "Output: ls\n"
            "Input: list all files with details\n"
            "Output: ls -la\n"
            "Input: remove file.txt\n"
            "Output: rm file.txt\n"
            "Input: create directory\n"
            "Output: mkdir\n"
            "Input: " + user + "\n"
            "Output: ";

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

        // Декодирование промпта
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            break;
        }

        // Генерация
        std::string response;
        int max_tokens = 30;
        
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
            
            // Останавливаемся на новой строке
            if (piece == "\n") {
                break;
            }
            
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;

            batch = llama_batch_get_one(&new_token, 1);
            if (llama_decode(ctx, batch) != 0) {
                break;
            }
        }
        printf("\n");
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
