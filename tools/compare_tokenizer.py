#!/usr/bin/env python3
"""Check the C tokenizer against Hugging Face's `tokenizers` library.

Generates a few thousand strings (hand-picked edge cases plus random text drawn
from many Unicode blocks), encodes each with both implementations using the
same tokenizer.json, and reports every difference.

    pip install tokenizers
    python3 tools/compare_tokenizer.py            # uses ./qwen3 and ./model
    python3 tools/compare_tokenizer.py --model DIR --binary PATH --n 20000
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer

EDGE_CASES = [
    "", " ", "  ", "   ", "\n", "\n\n", " \n", "\n ", "\t", " \t\n ", "\r\n", "a", " a", "a ", " a ", "a  b", "a   b",
    "Hello, world!", "Hello world", " Hello  world ", "It's", "it's I'm you're we've they'll I'd DON'T IT'S",
    "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D", "can't won't 'tis rock'n'roll", "'ſ", "x'ſy",
    "3.14159", "1,000,000", "2026-11-16", " 42", "a1b2c3", "v0.6B", "100%", "$5.00", "½ ² Ⅷ ٣",
    "foo(bar) { return x[i]; }", "a = b + c;\n\tif (x) {\n\t\ty();\n\t}\n", "!!!", "...", " ...", "...\n", "-->\n\n",
    "path/to/file.txt", "https://example.com/a?b=c&d=e", "user@example.com", "#include <stdio.h>",
    "def f(x):\n    return x ** 2\n\n\nprint(f(3))\n", "    indented", "\tTabbed", "trailing   ", "trailing\n\n\n",
    "你好，世界！", "日本語のテキスト", "한국어 텍스트",
    "Привет, мир!", "مرحبا بالعالم",
    "नमस्ते दुनिया", "שלום עולם",
    "café", "café", "Ångström", "Ångström", "Å Ω K", "é̂ ậ ậ",
    "각", "한글", "각", "क़ ড় יִ 豈",
    "\U0001f600", "\U0001f468‍\U0001f469‍\U0001f467", "\U0001f1ee\U0001f1f3 \U0001f44d\U0001f3fd", "a​b", "a b", "a　b",
    "a b c", "x\u0085y", "﻿BOM", "naïve résumé coöperate", "ß ẞ ﬁ ﬃ", "Ǆ ǅ ǆ",
    "The quick brown fox jumps over the lazy dog. " * 3, "a" * 300, " " * 50, "\n" * 20, "ab" * 200, "." * 100,
    "mixed中文and English和123数字", "สวัสดีครับ", "Ậ Ộ ộ",
]

# (first, last) code point ranges to draw random characters from, with weights
BLOCKS = [
    ((0x20, 0x7E), 30), ((0x09, 0x0D), 3), ((0x20, 0x20), 12), ((0x0A, 0x0A), 4), ((0x30, 0x39), 4), ((0x27, 0x27), 2),
    ((0xA0, 0xFF), 4), ((0x100, 0x24F), 3), ((0x300, 0x36F), 4), ((0x370, 0x3FF), 2), ((0x400, 0x4FF), 3),
    ((0x590, 0x6FF), 2), ((0x900, 0x97F), 2), ((0xE00, 0xE7F), 1), ((0x1100, 0x11FF), 2), ((0x1E00, 0x1FFF), 3),
    ((0x2000, 0x206F), 3), ((0x2070, 0x218F), 2), ((0x2190, 0x2BFF), 1), ((0x3000, 0x30FF), 3), ((0x4E00, 0x9FFF), 4),
    ((0xAC00, 0xD7A3), 3), ((0xF900, 0xFAFF), 1), ((0xFB00, 0xFB4F), 1), ((0xFF00, 0xFFEF), 1), ((0x1F300, 0x1F6FF), 2),
    ((0x1D400, 0x1D7FF), 1),
]
# Any code point at all, including unassigned and very recently added ones. Off by
# default: the reference library takes its character classes and its
# normalization data from two different Unicode versions, and neither is the one
# unicode_tables.h was generated from, so brand-new characters can legitimately differ.
EXOTIC_BLOCK = ((0x0, 0x2FFFF), 1)
WORDS = "the of and to in is that it's for I'm we'll don't you've they're he'd Hello world 2026 3.5 foo_bar x1 -- ... !? () [] {} \n \n\n \t   ".split(" ")


def random_string(rng):
    kind = rng.random()
    out = []
    if kind < 0.35:  # word salad: realistic spacing and punctuation
        for _ in range(rng.randint(1, 30)):
            out.append(rng.choice(WORDS))
            out.append(rng.choice([" ", " ", " ", "", "  ", "\n", ", ", ". "]))
    else:  # raw characters from weighted Unicode blocks
        total = sum(w for _, w in BLOCKS)
        for _ in range(rng.randint(1, 40)):
            r = rng.uniform(0, total)
            for (lo, hi), w in BLOCKS:
                r -= w
                if r <= 0:
                    break
            cp = rng.randint(lo, hi)
            if 0xD800 <= cp <= 0xDFFF or cp == 0:
                cp = 0x20
            ch = chr(cp)
            out.append(ch * rng.choice([1, 1, 1, 2, 3]))
    return "".join(out)


def c_encode(binary, model_dir, strings, parse_special):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(strings, f, ensure_ascii=bool(len(strings) % 2))  # exercise both raw UTF-8 and \u escapes
        path = f.name
    cmd = [binary, "-m", model_dir, "--encode-json", path] + (["--parse-special"] if parse_special else [])
    res = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(path)
    if res.returncode != 0:
        sys.exit("C tokenizer failed: " + res.stderr)
    lines = res.stdout.split("\n")[: len(strings)]
    return [[int(x) for x in line.split()] for line in lines]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model")
    ap.add_argument("--binary", default="./qwen3")
    ap.add_argument("--n", type=int, default=5000, help="number of random strings")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exotic", action="store_true", help="also draw from every code point, assigned or not")
    args = ap.parse_args()
    if args.exotic:
        BLOCKS.append(EXOTIC_BLOCK)

    hf = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    specials = [t.content for t in hf.get_added_tokens_decoder().values()]
    rng = random.Random(args.seed)

    plain = EDGE_CASES + [random_string(rng) for _ in range(args.n)]
    plain = [s for s in plain if not any(sp in s for sp in specials)]
    with_special = []
    for _ in range(max(200, args.n // 10)):
        parts = []
        for _ in range(rng.randint(1, 5)):
            parts.append(random_string(rng))
            parts.append(rng.choice(specials))
        with_special.append("".join(parts))
    with_special += ["<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_start|><|im_end|>",
                     "<think></think>", "a<|endoftext|>b", "<|im_start", "<|im_end|", "<<|im_end|>>"]

    bad = 0
    for name, strings, parse_special in [("plain text", plain, False), ("with special tokens", with_special, True)]:
        got = c_encode(args.binary, args.model, strings, parse_special)
        mismatches = 0
        for s, g in zip(strings, got):
            want = hf.encode(s, add_special_tokens=False).ids
            if g != want:
                mismatches += 1
                if mismatches <= 10:
                    print("MISMATCH on %r\n   C : %s\n   HF: %s" % (s[:80], g[:24], want[:24]))
        print("%-20s %6d strings, %d mismatches" % (name, len(strings), mismatches))
        bad += mismatches
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
