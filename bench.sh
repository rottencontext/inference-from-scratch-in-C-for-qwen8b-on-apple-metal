#!/bin/sh
# Measure the engine on one model and append a row to docs/_data/results.csv.
#
#   ./bench.sh                         Qwen3-8B in ./model-8b
#   ./bench.sh model-8b "bf16 matvec"  with a note saying what changed
#   ./bench.sh model                   another model directory (0.6B, for quick tries)
#   ./bench.sh --sweep                 thread scaling: one run per thread count,
#                                      rows go to docs/_data/sweep.csv
#
# Every run uses the same prompt, picks the top token every time (-t 0) and
# stops after a fixed number of tokens, so the work is identical between runs
# and the reply text can be compared with the saved one.
#
# What is recorded (the median of RUNS runs):
#   load_s         seconds to load the model
#   prompt_tok_s   speed of reading the prompt
#   ttft_s         load-free time to the first reply token (prompt tokens / speed)
#   gen_tok_s      speed of generating the reply: the headline number
#   peak_mem_gb    peak memory footprint of the process
#   swap_mb        how much the system's swap grew during the runs (should be 0)
#   logits_diff    largest difference from the saved baseline logits
#   same_text      whether the reply is identical to the saved baseline reply
#
# The first run on a model saves the baseline (logits and reply text) under
# bench/baseline/. Delete those files to start a new baseline.
#
# Settings, as environment variables:
#   RUNS=3  NEW_TOKENS=64  THREADS=(engine default)  SWEEP_THREADS="1 2 4 6 12 18"
set -e
cd "$(dirname "$0")"

PROMPT="Explain how a transformer language model generates text."
# Token ids of "The capital of France is Paris, and the capital of Japan is".
# The logits after each of these 13 tokens are what gets compared.
LOGIT_IDS="785,6722,315,9625,374,12095,11,323,279,6722,315,6323,374"

SWEEP=0
if [ "${1:-}" = "--sweep" ]; then
    SWEEP=1
    shift
fi
MODEL="${1:-model-8b}"
NOTE="${2:-}"
RUNS="${RUNS:-3}"
NEW_TOKENS="${NEW_TOKENS:-64}"
SWEEP_THREADS="${SWEEP_THREADS:-1 2 4 6 12 18}"

case "$NOTE" in
    *,*|*\"*) echo "error: the note cannot contain a comma or a double quote (it goes into a CSV)" >&2; exit 1 ;;
esac
if [ ! -f "$MODEL/config.json" ]; then
    echo "error: no model in $MODEL/ (run ./download_model.sh or ./run_8b.sh --download-only)" >&2
    exit 1
fi

if [ ! -x ./qwen3 ] || [ qwen3.c -nt ./qwen3 ]; then
    ./build.sh
fi
if [ ! -x tests/cmp_f32 ] || [ tests/cmp_f32.c -nt tests/cmp_f32 ]; then
    ${CC:-cc} -O2 -o tests/cmp_f32 tests/cmp_f32.c -lm
fi

NAME="$(basename "$MODEL")"
BASE_DIR="bench/baseline"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$BASE_DIR" docs/_data

HEADER="date,commit,model,threads,runs,new_tokens,load_s,prompt_tok_s,ttft_s,gen_tok_s,peak_mem_gb,swap_mb,logits_diff,same_text,note"

# The commit being measured; a "+" marks uncommitted changes to the engine.
commit="$(git rev-parse --short HEAD 2>/dev/null || echo none)"
if ! git diff --quiet HEAD -- qwen3.c build.sh unicode_tables.h 2>/dev/null; then
    commit="$commit+"
fi

swap_used_mb() {
    if [ "$(uname -s)" = "Darwin" ]; then
        sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/'
    else
        awk '/SwapTotal/ { t = $2 } /SwapFree/ { f = $2 } END { printf "%.2f", (t - f) / 1024 }' /proc/meminfo
    fi
}

# Middle value of the numbers on stdin (one per line).
median() {
    sort -n | awk '{ v[NR] = $1 } END { if (NR) print v[int((NR + 1) / 2)] }'
}

# One timed run. Writes the reply to $TMP/reply.txt and one line of numbers,
# "load_s prompt_tokens prompt_tok_s gen_tok_s peak_mem_gb", to stdout.
one_run() {
    threads_opt=""
    if [ -n "$1" ]; then threads_opt="--threads $1"; fi
    # shellcheck disable=SC2086
    /usr/bin/time -l ./qwen3 -m "$MODEL" -p "$PROMPT" --no-think -t 0 --seed 1 \
        -n "$NEW_TOKENS" --stats $threads_opt > "$TMP/reply.txt" 2> "$TMP/err.txt" || {
        cat "$TMP/err.txt" >&2
        echo "error: the engine failed" >&2
        exit 1
    }
    awk '
        /^loaded / { for (i = 1; i <= NF; i++) if ($i == "in") { load = $(i + 1); sub(/s:$/, "", load) } }
        /^\[prompt: / { ptok = $2; ps = $4; gs = $10 }
        /maximum resident set size/ { rss = $1 }
        /peak memory footprint/ { foot = $1 }
        END {
            mem = foot ? foot : rss
            printf "%s %s %s %s %.2f\n", load, ptok, ps, gs, mem / 1073741824
        }' "$TMP/err.txt"
}

# RUNS timed runs with the given thread count ("" = engine default); sets the
# variables that make up a CSV row.
measure() {
    : > "$TMP/runs.txt"
    i=1
    while [ "$i" -le "$2" ]; do
        line="$(one_run "$1")"
        echo "  run $i/$2: $(echo "$line" | awk '{ printf "load %ss, prompt %s tok/s, generate %s tok/s, peak %s GB", $1, $3, $4, $5 }')" >&2
        echo "$line" >> "$TMP/runs.txt"
        i=$((i + 1))
    done
    load_s="$(awk '{ print $1 }' "$TMP/runs.txt" | median)"
    prompt_tokens="$(awk 'NR == 1 { print $2 }' "$TMP/runs.txt")"
    prompt_tok_s="$(awk '{ print $3 }' "$TMP/runs.txt" | median)"
    gen_tok_s="$(awk '{ print $4 }' "$TMP/runs.txt" | median)"
    peak_mem_gb="$(awk '{ print $5 }' "$TMP/runs.txt" | median)"
    ttft_s="$(awk -v n="$prompt_tokens" -v s="$prompt_tok_s" 'BEGIN { if (s > 0) printf "%.2f", n / s }')"
    threads="$(sed -n 's/.* \([0-9]*\) threads$/\1/p' "$TMP/err.txt" | head -1)"
}

append_row() {
    file="$1"
    if [ ! -f "$file" ]; then echo "$HEADER" > "$file"; fi
    echo "$(date +%Y-%m-%d),$commit,$NAME,$threads,$2,$NEW_TOKENS,$load_s,$prompt_tok_s,$ttft_s,$gen_tok_s,$peak_mem_gb,$swap_mb,$logits_diff,$same_text,$NOTE" >> "$file"
}

# ---- thread scaling: one run per thread count, no correctness check ----------
if [ "$SWEEP" = "1" ]; then
    echo "thread sweep on $NAME ($commit): $SWEEP_THREADS"
    for t in $SWEEP_THREADS; do
        swap0="$(swap_used_mb)"
        echo "threads $t" >&2
        measure "$t" 1
        swap_mb="$(awk -v a="$swap0" -v b="$(swap_used_mb)" 'BEGIN { d = b - a; printf "%.0f", (d > 0) ? d : 0 }')"
        logits_diff=""
        same_text=""
        append_row docs/_data/sweep.csv 1
    done
    echo
    echo "threads  generate tok/s"
    tail -n "$(echo "$SWEEP_THREADS" | wc -w | tr -d ' ')" docs/_data/sweep.csv | awk -F, '{ printf "%7s  %s\n", $4, $10 }'
    exit 0
fi

# ---- the normal benchmark ------------------------------------------------------
echo "benchmarking $NAME ($commit): $RUNS runs of $NEW_TOKENS new tokens"
swap0="$(swap_used_mb)"
measure "${THREADS:-}" "$RUNS"
swap_mb="$(awk -v a="$swap0" -v b="$(swap_used_mb)" 'BEGIN { d = b - a; printf "%.0f", (d > 0) ? d : 0 }')"

# Reply text against the saved one.
if [ -f "$BASE_DIR/$NAME.txt" ]; then
    if cmp -s "$TMP/reply.txt" "$BASE_DIR/$NAME.txt"; then same_text="yes"; else same_text="no"; fi
else
    cp "$TMP/reply.txt" "$BASE_DIR/$NAME.txt"
    same_text="baseline"
fi

# Logits against the saved ones.
echo "  logits for ${LOGIT_IDS%%,*},... (13 positions)" >&2
./qwen3 -m "$MODEL" -c 512 --dump-logits "$LOGIT_IDS" "$TMP/logits.bin" 2> /dev/null
if [ -f "$BASE_DIR/$NAME.bin" ]; then
    # Tolerance 1e9: this only reports the difference; judging it is up to you.
    logits_diff="$(./tests/cmp_f32 "$TMP/logits.bin" "$BASE_DIR/$NAME.bin" 1e9 | sed -n 's/.*largest difference \([^ ]*\).*/\1/p')"
    if [ -z "$logits_diff" ]; then logits_diff="size-mismatch"; fi
else
    cp "$TMP/logits.bin" "$BASE_DIR/$NAME.bin"
    logits_diff="baseline"
fi

append_row docs/_data/results.csv "$RUNS"

echo
echo "model            $NAME ($commit, $threads threads)"
echo "load             $load_s s"
echo "prompt           $prompt_tok_s tok/s  ($prompt_tokens tokens, first token after $ttft_s s)"
echo "generate         $gen_tok_s tok/s"
echo "peak memory      $peak_mem_gb GB"
echo "swap growth      $swap_mb MB"
echo "logits vs base   $logits_diff"
echo "reply vs base    $same_text"
echo
echo "row added to docs/_data/results.csv"
