# Llama Dynamic Ctx

**Dynamic context management for LLMs**
...because humans don't hit 'context limit exceeded' and start forgetting your dog's name. Let's make this stuff uncanny as hell <3

## Quick Start

```bash
git clone https://github.com/vifirsanova/llama-dynamic-context
cd llama-dynamic-context
mkdir build && cd build
cmake .. -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=OFF
make random-trimming -j$(nproc)
```

## Example Usage

```bash
# 15% KV-cache trimming
./bin/random-trimming -m models/your-model.gguf -trim_pct 15
```

## Output Preview

```
[KV_CACHE_STATE] Utilization: 49% → trimming 15%...
[DELETED_TOKENS_REPORT] Removed 151 tokens
TRIMMED TOKEN DISTRIBUTION:
 - Early: 48 tokens (31.8%)
 - Middle: 59 tokens (39.1%)
 - Late: 44 tokens (29.1%)
```

[Full output example](logs/random-trimming-output.txt)

## Features

- Continuous KV-cache trimming during generation
- Real-time cache utilization tracking
- Visual cache state debugging
- Mobile and desktop optimized

## Tested Setup

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

## Building Notes

If build fails: `rm -rf build && mkdir build` usually fixes it

## Contributing

PRs welcome

## Distribution & Usage

-> do whatever the fuck you want with it
