---
layout: post
title: "Baseline: 7.3 tokens per second"
commit: f3947aa
tok_s: 7.3
walkthrough: /walkthroughs/baseline/
summary: "The unoptimized engine on Qwen3-8B: 7.3 tokens/s, 30.6 GB of memory, and two numbers that look wrong."
---

This is the starting point. Nothing has been optimized yet; every later report is
measured against these numbers.

## The setup

- **Engine:** `qwen3.c`, one C file with no libraries. It converts the model's 16-bit
  weights to 32-bit floats when it loads them and runs on 6 threads, one per fast core.
- **Model:** Qwen3-8B, 36 layers, 8.2 billion weights, 16.4 GB on disk.
- **Machine:** MacBook Pro, Apple M5 Pro (6 + 12 cores), 48 GB of memory.
- **Benchmark:** `./bench.sh`. One fixed prompt of 22 tokens, 64 generated tokens, the
  top token picked every time so the reply is always the same. Three runs, median reported.

## Numbers

| | Measured |
|---|---|
| Generate (tokens/s) | 7.3 |
| Prompt (tokens/s) | 2.0 |
| Time to first token (s) | 11.0 |
| Load (s) | 24.6 |
| Peak memory (GB) | 30.58 |
| Swap growth (MB) | 3916 |

The three runs generated at 7.5, 7.3 and 7.0 tokens/s. The first load took 31.0 s and
the next two 24.6 s each.

The reply and the output scores from this run are saved as the reference. Every later
change is checked against them, so a speed-up that changes the model's answers shows up
in the same table as the speed-up.

## Two things that look wrong

**Reading the prompt is slower than generating.** The prompt went through at 2.0
tokens/s and the reply at 7.3. Both do the same work per token, and on the small 0.6B
model the prompt is the faster of the two (115 against 83 tokens/s). So something makes
the first tokens after loading slow on 8B. That costs 11 seconds before the first word
appears.

**The Mac's swap grew by 3.9 GB during the benchmark.** A 30.6 GB process should fit in
48 GB without swapping. If macOS is pushing the model's memory out while it loads, that
would also explain the slow first tokens, but that is a guess: it has not been measured.

Until the swap is gone, the numbers above are a floor, not a clean measurement.

## Next

1. Find out why the machine swaps, and whether the slow prompt is the same problem.
2. Measure how fast this Mac can read memory. Each token reads about 30 GB of weights,
   so that speed sets the ceiling for the current design.
3. Run the thread sweep (`./bench.sh --sweep`) and a profile, to see whether the engine
   is limited by arithmetic or by memory.
