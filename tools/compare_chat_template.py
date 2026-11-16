#!/usr/bin/env python3
"""Check the C chat loop against the official Qwen3 chat template.

The C engine builds the conversation incrementally and rewinds its KV cache to
rewrite history. This script replays a scripted multi-turn conversation through
it, records every token it puts in the context (--log-tokens), and compares the
final context with what the Jinja chat template + Hugging Face tokenizer
produce for the same conversation. It works with any weights, including the
random test model, because the replies are scripted (--replay).

    pip install tokenizers jinja2
    python3 tools/compare_chat_template.py                     # template from model/tokenizer_config.json
    python3 tools/compare_chat_template.py --template file.jinja --model DIR
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

USERS = ["What is 2+2?", "And times three?  (quick)", "Now explain — résumé it's", "one more", "last one", "bye"]
REPLIES = [
    "<think>\nSimple arithmetic.\n</think>\n\nThe answer is 4.",   # normal thinking reply
    "No thinking block here, just an answer.",                     # direct answer
    "<think>\n\n</think>\n\nTwelve.",                               # empty thinking block
    "<think>\nfirst</think>mid<think>\nsecond\n</think>\n\n\n\nAfter two blocks.",  # only the last </think> counts
    "<think>\nthis block is never closed",                         # cut off while thinking
    "Final.",
]


def run_c(binary, model, users, replies, think, system):
    with tempfile.TemporaryDirectory() as d:
        log, rep = os.path.join(d, "log.txt"), os.path.join(d, "replay.txt")
        with open(rep, "w") as f:
            f.write("\n".join(" ".join(map(str, r)) for r in replies))
        cmd = [binary, "-m", model, "--log-tokens", log, "--replay", rep, "-c", "2048", "--think" if think else "--no-think"]
        if system:
            cmd += ["-s", system]
        res = subprocess.run(cmd, input="\n".join(users) + "\n", capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit("C engine failed: " + res.stderr)
        ctx, end = {}, 0
        for line in open(log):
            pos, tok = map(int, line.split())
            ctx[pos] = tok   # a rewind overwrites earlier entries
            end = pos + 1
        return [ctx[i] for i in range(end)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model")
    ap.add_argument("--binary", default="./qwen3")
    ap.add_argument("--template", default=None, help="Jinja chat template file (default: read tokenizer_config.json)")
    args = ap.parse_args()

    if args.template:
        template_src = open(args.template, encoding="utf-8").read()
    else:
        template_src = json.load(open(os.path.join(args.model, "tokenizer_config.json"), encoding="utf-8"))["chat_template"]
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(template_src)
    hf = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    reply_ids = [hf.encode(r, add_special_tokens=False).ids for r in REPLIES]

    failures = 0
    for think in (True, False):
        for system in (None, "You are terse."):
            for turns in range(1, len(USERS) + 1):
                got = run_c(args.binary, args.model, USERS[:turns], reply_ids[:turns], think, system)
                messages = [{"role": "system", "content": system}] if system else []
                for i in range(turns):
                    messages.append({"role": "user", "content": USERS[i]})
                    if i < turns - 1:
                        messages.append({"role": "assistant", "content": REPLIES[i]})
                text = template.render(messages=messages, add_generation_prompt=True, enable_thinking=think)
                want = hf.encode(text, add_special_tokens=False).ids + reply_ids[turns - 1]
                ok = got == want
                failures += not ok
                print("%-9s %-10s %d turn(s): %s (%d tokens)" % ("think" if think else "no-think", "system" if system else "no system",
                                                                turns, "ok" if ok else "MISMATCH", len(want)))
                if not ok:
                    k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
                    print("   first difference at token %d\n   C   : %s\n   want: %s" % (k, got[max(0, k - 4):k + 8], want[max(0, k - 4):k + 8]))
    print("RESULT: %s" % ("MATCH" if not failures else "%d MISMATCHES" % failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
