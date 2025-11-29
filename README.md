# Dynamic context management for LLMs

...because humans don't hit 'context limit exceeded'


This is my **standalone fork** of llama.cpp focused specifically on dynamic context management

## Core Modifications

**Added Files:**
- `examples/random-trimming/random-trimming.cpp` - Main chat with continuous trimming
- `examples/command-inference/command-inference.cpp` - **NEW: English to Linux command converter**
- `examples/random-trimming/CMakeLists.txt` - Build configuration
- `examples/command-inference/CMakeLists.txt` - **NEW: Build config for command inference**
- `logs/random-trimming-output.txt` - Full example conversation with trimming logs

**Major Additions to llama-kv-cache.cpp:**
- **Token Mapping** - Real-time position-to-token tracking
- **Memory Analytics** - Сache utilization calculations
- **Cell Management** - Direct cell eviction with verification
- **Trim Algorithms** - Percentage-based random trimming **with uniform distribution** ensures even token removal across the entire sequence to prevent localized information loss
- **Debug Integration** - External position tracking hooks

**API Changes:**
- New `trim_random()` method in KV-cache with percentage-based trimming
- Additional debugging and visualization hooks throughout the stack
- Modified generation loop for continuous context optimization

**Preserved Compatibility:**
- All original model formats (GGUF)
- Existing CPU/GPU backends
- Basic inference API surface
- Core sampling algorithms

**Key Technical Changes:**
```cpp
// Before (upstream):
// [prompt] → [generate] → [context_full?] → [error]

// After (this fork):
// [prompt] → [trim if needed] → [generate] → [repeat]
```

## Dataset

**NEW: LinLM Dataset** 

A curated synthetic dataset for Linux command inference testing: [![HF Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/missvector/linux-commands)

* 10+ languages
* Arch Linux commands recognition
* Fine-tune and test LLM for development, system administration, file operations, Git, Docker, and more

## Quick Start

```bash
git clone https://github.com/vifirsanova/llama-dynamic-context
cd llama-dynamic-context
mkdir build && cd build
cmake .. -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=OFF

# Build chat with trimming
make random-trimming -j$(nproc)

# Build command inference tool
make command-inference -j$(nproc)
```

## Example Usage

```bash
# 15% KV-cache trimming
./bin/random-trimming -m models/your-model.gguf -trim_pct 15

# English to Linux command converter
./bin/command-inference -m models/your-model.gguf
```

## Random Trimming 

**Output Preview**

```
[KV_CACHE_STATE] Utilization: 49% → trimming 15%...
[DELETED_TOKENS_REPORT] Removed 151 tokens
TRIMMED TOKEN DISTRIBUTION:
 - Early: 48 tokens (31.8%)
 - Middle: 59 tokens (39.1%)
 - Late: 44 tokens (29.1%)
```

[Full output example](logs/random-trimming-output.txt)

**Features**

- Continuous KV-cache trimming during generation
- Uniform distribution sampling - prevents localized information loss by evenly distributing trimmed tokens across the entire context
- Real-time cache utilization tracking
- Visual cache state debugging
- Mobile and desktop optimized

**Test Setup:**
- **Model:** Phi-3-mini-4k-instruct-q4.gguf
- **Hardware:** Intel i9 + RTX 4090
- **Context:** 2048 tokens
- **Trim:** 15% continuous

| Metric | Value | Notes |
|--------|-------|-------|
| **Speed** | 16-20 tokens/sec | Consistent during 10+ turn dialogue |
| **Memory Usage** | ~1.4GB | Peak during 49% cache utilization |
| **Tokens Processed** | 1000+ | Full conversation length |
| **Trimming Operations** | 7 | Automatic during generation |
| **Tokens Removed** | ~700 total | 85% context retention rate |
| **Cache Efficiency** | 35-45% density | After compaction |
| **Context Quality** | Maintained facts | Remembered pets, names, details |

*Measured from actual logs - see `examples/logs/random-trimming-output.txt`*

## Command Inference

**NEW:** Real-time English to Linux command converter using few-shot learning:

```bash
> show all files in directory
ls -la
> create new folder
mkdir
> delete all files  
rm -rf *
> show current user
whoami
```

**Features:**
- Zero configuration setup
- Few-shot learning with curated examples
- Handles complex commands (`cd folder && touch file.txt`)
- Clean output format

## Building Notes

If build fails: `rm -rf build && mkdir build` usually fixes it

## Contributing

PRs welcome

## Citation

```
@misc{llamadynamiccontext2025,
  author = {V. Firsanova},
  title = {Llama Dynamic Context},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/vifirsanova/llama-dynamic-context/}}
}
```
