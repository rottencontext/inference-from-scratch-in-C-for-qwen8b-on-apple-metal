#!/usr/bin/env python3
"""Create a small Qwen3 model with random weights, for testing the C engine.

The real checkpoint is 1.2 GB; this one is a few megabytes but has exactly the
same architecture, so any mistake in the C forward pass (normalization order,
RoPE layout, grouped-query attention, weight tying, dtype conversion) shows up
as a logit mismatch against the reference implementation in `transformers`.

    pip install torch transformers safetensors
    python3 tools/make_test_model.py OUT_DIR [--dtype bf16|f16|f32] [--untied] [--new-config]

Copy a tokenizer.json into OUT_DIR if you also want to chat with it (it will
only produce noise, of course).
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"])
    ap.add_argument("--untied", action="store_true", help="separate lm_head instead of sharing the embedding")
    ap.add_argument("--new-config", action="store_true", help="write rope_theta inside rope_parameters (newer transformers)")
    ap.add_argument("--full-size", action="store_true", help="use the real Qwen3-0.6B dimensions (1.2 GB file)")
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # Deliberately awkward sizes: head_dim != hidden_size / heads, 3 query heads
    # per key/value head, widths that are not multiples of the SIMD block.
    cfg = dict(vocab_size=args.vocab, hidden_size=100, intermediate_size=172, num_hidden_layers=3, num_attention_heads=6,
               num_key_value_heads=2, head_dim=24, max_position_embeddings=2048, rms_norm_eps=1e-6,
               tie_word_embeddings=not args.untied)
    if args.full_size:
        cfg.update(hidden_size=1024, intermediate_size=3072, num_hidden_layers=28, num_attention_heads=16,
                   num_key_value_heads=8, head_dim=128, max_position_embeddings=40960)
    config = Qwen3Config(**cfg, rope_parameters={"rope_type": "default", "rope_theta": 1000000.0})
    model = Qwen3ForCausalLM(config)

    # Random weights everywhere, including the norm gains (which default to 1
    # and would hide a missing multiplication).
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            elif "embed_tokens" in name or "lm_head" in name:
                p.copy_(0.5 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / (p.shape[1] ** 0.5) * 1.5)

    dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[args.dtype]
    state = {k: v.to(dtype).contiguous() for k, v in model.state_dict().items()}
    if not args.untied:
        state.pop("lm_head.weight", None)  # tied checkpoints store the shared matrix once

    os.makedirs(args.out, exist_ok=True)
    save_file(state, os.path.join(args.out, "model.safetensors"))

    out_cfg = {"architectures": ["Qwen3ForCausalLM"], "attention_bias": False, "hidden_act": "silu", "model_type": "qwen3",
               "torch_dtype": {"bf16": "bfloat16", "f16": "float16", "f32": "float32"}[args.dtype], **cfg}
    if args.new_config:
        out_cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": 1000000.0}
    else:
        out_cfg["rope_theta"] = 1000000
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
    n = sum(v.numel() for v in state.values())
    print("wrote %s: %.1fM parameters, %s, %s" % (args.out, n / 1e6, args.dtype, "untied" if args.untied else "tied"))


if __name__ == "__main__":
    main()
