# qwen3-c

A complete inference engine for Qwen3 language models in one C file. It reads the files
Hugging Face ships (`config.json`, `tokenizer.json`, `model.safetensors`) and runs the
model with no Python, no ML framework and no BLAS library. Threads come from pthreads,
SIMD from compiler vector types, and everything else is in `qwen3.c`.

It targets **Qwen3-0.6B**, and reads every dimension from `config.json`, so the larger
dense Qwen3 models load too. The MoE variants (the `-A3B` / `-A22B` models) do not.

## Quick start

```sh
./download_model.sh        # about 1.5 GB from huggingface.co into ./model
./build.sh                 # builds ./qwen3
./build.sh test            # self-test (see "How it was checked")

./qwen3 -p "Why is the sky blue?"            # one answer, then exit
./qwen3                                      # interactive chat
./qwen3 --no-think -s "Answer in one line."  # no reasoning block, with a system prompt
```

`build.sh` is a plain shell script: one compiler command, with `CC=` and `CFLAGS=` as
overrides. The equivalent by hand is `cc -O3 -o qwen3 qwen3.c -lm -lpthread`.

## Status

Read this before trusting it.

| What | State |
|---|---|
| Forward pass | Matches Hugging Face `transformers` to about 1e-6 relative error on random-weight models, including one with the exact Qwen3-0.6B dimensions |
| Tokenizer | Identical to Hugging Face `tokenizers` on over 150,000 test strings, using a vocabulary and merge list that are byte-identical (SHA-256) to the real Qwen3-0.6B `tokenizer.json` |
| Chat template | Identical context to the model's own Jinja template over 24 scripted conversations |
| Real file layout | The real `config.json` and the header of the real `model.safetensors` were read: all 311 tensors have the names, shapes and dtype (BF16) the loader expects |
| Memory safety | Clean under AddressSanitizer, UndefinedBehaviorSanitizer and ThreadSanitizer |
| **Real Qwen3-0.6B weights** | **Never loaded or run.** The sandboxes this was built in could not download from huggingface.co, so no real text has been generated with it yet |
| **macOS** | **Not built or run on macOS.** Built and tested on Linux, x86-64 (gcc 13, clang 18) and arm64 (gcc 11) |

So the first thing to do after downloading is `./build.sh test`, then one real prompt. If the
model answers sensibly, the last two rows are settled. For a stronger check against the
real weights:

```sh
pip install torch transformers safetensors tokenizers jinja2
python3 tools/compare_logits.py          # logits vs transformers, real weights
python3 tools/compare_tokenizer.py       # tokenizer vs Hugging Face, thousands of strings
python3 tools/compare_chat_template.py   # chat loop vs the official template
```

## Options

```
-m DIR            model directory                               (default: model)
-p TEXT           answer one prompt and exit; without it, chat
-s TEXT           system prompt
-n N              max new tokens per reply                      (default: 2048)
-c N              context window in tokens                      (default: 4096)
--think           reason first                                  (default)
--no-think        answer directly
-t F              temperature, 0 = greedy                       (default: 0.6, or 0.7 with --no-think)
-k N              top-k, 0 = off                                (default: 20)
-P F              top-p                                         (default: 0.95, or 0.8 with --no-think)
--presence-penalty F   try 1.0 to 1.5 if a reply starts looping (default: 0)
--seed N          fixed random seed
--threads N       worker threads                                (default: performance cores)
--raw             no chat template; continue the text given with -p
--stats           print tokens per second after each reply
```

In chat, `/reset` clears the conversation and `/exit` quits. Ctrl-C stops the current
reply; at the prompt it exits.

The sampling defaults are the ones the Qwen3 model card recommends for each mode. The
card also advises against greedy decoding (`-t 0`) in thinking mode.

Memory: about 2.4 GB for the weights (stored as float32) plus 0.23 MB per token of
context for the KV cache, so roughly 3.4 GB at the default 4096-token window. The file
on disk is 1.5 GB because it stores the output matrix separately from the embedding
table; the two are identical, so only one copy is loaded.

## How it works

`qwen3.c` is laid out in the order data moves through the model. Each section opens
with a comment explaining the idea before the code.

| Section | What it does |
|---|---|
| 1. Utilities | Memory, file reading, timing |
| 2. JSON parser | Recursive descent into an arena-allocated tree. All three input files start with JSON |
| 3. Unicode | UTF-8 decoding, letter/number/space classes, NFC normalization |
| 4. Tokenizer | Byte-level BPE: normalize, split into word-like pieces, merge byte pairs by rank |
| 5. Safetensors loader | `mmap` the file, parse the header, widen BF16 weights to float32 |
| 6. Thread pool and math | A small work-sharing pool and the SIMD dot product everything is built on |
| 7. The model | RMSNorm, grouped-query attention with QK-norm and RoPE, SwiGLU MLP, KV cache |
| 8. Sampler | Temperature, top-k, top-p |
| 9. Chat loop and CLI | The ChatML prompt format, thinking mode, history, generation |

One token's trip through the model, with Qwen3-0.6B's numbers:

```
token id
  -> embedding row                              1,024 numbers
  -> 28 layers, each:
       x += attention(rmsnorm(x))               16 query heads sharing 8 key/value heads
       x += mlp(rmsnorm(x))                     expands to 3,072 and back
  -> rmsnorm -> output matrix                   151,936 scores, one per vocabulary entry
  -> sampler                                    next token id
```

Three details differ from a plain Llama-style model and are easy to get wrong:

- **QK-norm.** Each query head and key head gets its own RMSNorm (over the 128 numbers
  of that head) before the rotary rotation.
- **RoPE layout.** Element `i` is paired with element `i + 64`, not with `i + 1`.
- **Explicit `head_dim`.** 16 heads x 128 = 2,048, which is twice the model width, so
  the query and output projections are not square.

### The chat loop and the KV cache

The engine feeds one token at a time and keeps every layer's keys and values, so each
new token costs one pass through the layers. Conversation state is just "how many cache
positions are valid". That makes two things cheap:

- `/reset` sets the position to zero.
- The official template stores past assistant turns without their `<think>` block. The
  cache holds the reply as generated, thinking included, so before the next turn the
  engine rewinds to the start of that reply and feeds the answer part again.

The second point costs a re-run of the previous answer at the start of each turn. It is
what the template specifies, and it keeps long reasoning out of the context window.

## How it was checked

`./build.sh test` runs two checks that need nothing but a C compiler:

1. **Forward pass.** `tests/tiny` is a 0.4M-parameter model with random weights and the
   Qwen3 architecture, using awkward sizes on purpose (3 query heads per key/value head,
   `head_dim` different from width/heads, widths that are not multiples of 8). The
   engine's logits for 300 tokens are compared with logits computed by `transformers`.
2. **Tokenizer.** 489 strings are compared with ids from Hugging Face `tokenizers`. This
   needs the real `tokenizer.json`, so it is skipped until the model is downloaded.

The `tools/` scripts are what produced the Status table:

| Script | Checks |
|---|---|
| `compare_logits.py` | Logits at every position vs `transformers` in float32. The reference runs the whole sequence at once with no cache, so this also validates the KV cache |
| `compare_tokenizer.py` | Token ids vs `tokenizers` on edge cases and random text from many scripts |
| `compare_chat_template.py` | The exact token context after scripted multi-turn chats vs the Jinja template |
| `make_test_model.py` | Creates the random-weight test models (`--full-size` for the real dimensions) |
| `gen_unicode_tables.py` | Regenerates `unicode_tables.h` from Python's `unicodedata` |

The tokenizer was tested with a `tokenizer.json` rebuilt from the Qwen vocabulary and
merge list in the llama.cpp repository, plus Qwen3's added tokens. Its vocabulary and
merges hash to the same SHA-256 as the real file's, its added tokens are the same 26,
and it reproduces all 47 of llama.cpp's own Qwen test vectors.

## Known limits

- **Speed.** Correctness came first. The weights are float32 and tokens are processed one
  at a time, so prompt processing is no faster than generation.
- **Unicode versions.** Character classes and normalization come from Unicode 15.1. The
  reference tokenizer uses two other Unicode versions for those two jobs, so a handful
  of characters added to Unicode very recently can tokenize differently. Run
  `compare_tokenizer.py --exotic` to see them.
- **History and `</think>`.** The template drops newlines after `</think>` at the text
  level; the engine drops whole newline-only tokens. They agree on everything the model
  normally produces.
- **Platform.** POSIX only (`mmap`, pthreads). No Windows build.
- **Features.** No tool calling, no batching, no quantization, no GPU.

## Where to take it next

Roughly in order of speed gained per line of code:

1. **Keep weights as 16-bit in memory** and widen inside the dot product. Generation is
   limited by how fast memory can be read, and this halves the bytes per token.
2. **Batch the prompt.** Process all prompt tokens through each layer together, turning
   matrix-vector products into matrix-matrix ones.
3. **4-bit or 8-bit quantization**, the main reason llama.cpp is fast on laptops.
4. **Metal** for the matrix products, which is the path toward what MLX does.

## Files

```
qwen3.c             the engine
unicode_tables.h    generated Unicode data (letter/number ranges, NFC tables)
build.sh            ./build.sh, ./build.sh test, ./build.sh debug
download_model.sh   fetches the model files with curl
tests/              tiny model, reference logits, tokenizer vectors, cmp_f32.c
tools/              the verification scripts described above
```

The model weights are Alibaba's, released under Apache 2.0.
