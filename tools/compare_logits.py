#!/usr/bin/env python3
"""Check the C engine's numbers against the reference implementation.

Runs the same token sequence through `transformers` (in float32, all positions
in one pass, no cache) and through ./qwen3 (one token at a time, with the KV
cache), then compares the logits at every position.

    pip install torch transformers safetensors
    python3 tools/compare_logits.py                      # ./qwen3, ./model, a built-in prompt
    python3 tools/compare_logits.py --model DIR --text "any prompt"
    python3 tools/compare_logits.py --model DIR --random 400   # 400 random token ids

Typical result for float32 arithmetic: differences around 1e-5 to 1e-4, and the
most likely next token identical at every position.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from transformers import AutoModelForCausalLM

DEFAULT_TEXT = ("<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n"
                "<think>\n\n</think>\n\nSunlight is scattered by air molecules, and blue light scatters the most.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model")
    ap.add_argument("--binary", default="./qwen3")
    ap.add_argument("--text", default=None)
    ap.add_argument("--random", type=int, default=0, help="use this many random token ids instead of text")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:  # the keyword was renamed in transformers 5
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    vocab = model.config.vocab_size

    if args.random:
        rng = np.random.default_rng(args.seed)
        ids = rng.integers(0, vocab, size=args.random).tolist()
    else:
        text = args.text or DEFAULT_TEXT
        out = subprocess.run([args.binary, "-m", args.model, "--encode", text, "--parse-special"], capture_output=True, text=True)
        if out.returncode != 0:
            sys.exit("C tokenizer failed: " + out.stderr)
        ids = [int(x) for x in out.stdout.split()]
    print("%d tokens" % len(ids))

    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0].float().numpy()
    del model  # free the reference copy of the weights before the C engine loads its own

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    cmd = [args.binary, "-m", args.model, "-c", str(max(len(ids), 16)), "--dump-logits", ",".join(map(str, ids)), path]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("C engine failed: " + res.stderr)
    got = np.fromfile(path, dtype=np.float32).reshape(len(ids), vocab)
    os.unlink(path)

    diff = np.abs(got - ref)
    scale = np.abs(ref).max()
    same_top = int((got.argmax(-1) == ref.argmax(-1)).sum())
    print("largest logit in the reference : %.3f" % scale)
    print("max  |C - reference|            : %.3g  (%.2g relative)" % (diff.max(), diff.max() / scale))
    print("mean |C - reference|            : %.3g" % diff.mean())
    print("worst position                  : %d" % int(diff.max(-1).argmax()))
    print("same top-1 token                : %d / %d positions" % (same_top, len(ids)))
    ok = diff.max() / scale < 2e-3 and same_top >= len(ids) - max(1, len(ids) // 100)
    print("RESULT: %s" % ("MATCH" if ok else "MISMATCH"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
