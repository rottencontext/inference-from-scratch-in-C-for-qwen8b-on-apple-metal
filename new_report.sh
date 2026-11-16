#!/bin/sh
# Start a blog report for the change just measured with ./bench.sh.
#
#   ./new_report.sh "Keep the weights in 16 bits"
#   ./new_report.sh "Keep the weights in 16 bits" model     for another model
#
# Writes docs/_posts/DATE-title.md with a before/after table filled in from
# the last two rows for that model in docs/_data/results.csv. Then write the
# prose, commit, and push: GitHub Pages publishes it. The report becomes one
# block on the home page; its page shows the numbers and, if the front matter
# names one, the walkthrough for that iteration.
set -e
cd "$(dirname "$0")"

TITLE="${1:-}"
MODEL="${2:-model-8b}"
CSV="docs/_data/results.csv"
if [ -z "$TITLE" ]; then
    echo "usage: $0 \"Title of the report\" [model-dir]" >&2
    exit 1
fi
if [ ! -f "$CSV" ]; then
    echo "error: $CSV does not exist; run ./bench.sh first" >&2
    exit 1
fi

slug="$(printf '%s' "$TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\{1,\}/-/g; s/^-//; s/-$//')"
post="docs/_posts/$(date +%Y-%m-%d)-$slug.md"
if [ -f "$post" ]; then
    echo "error: $post already exists" >&2
    exit 1
fi

# Columns: 2 commit, 4 threads, 7 load_s, 8 prompt_tok_s, 9 ttft_s, 10 gen_tok_s,
#          11 peak_mem_gb, 12 swap_mb, 13 logits_diff, 14 same_text
awk -F, -v model="$(basename "$MODEL")" -v title="$TITLE" '
    function ratio(a, b) { return (a > 0) ? sprintf("%.2f×", b / a) : "" }
    function line(label, col, change) {
        if (n > 1) printf "| %s | %s | %s | %s |\n", label, prev[col], cur[col], change
        else printf "| %s | %s |\n", label, cur[col]
    }
    NR > 1 && $3 == model { n++; for (i = 1; i <= NF; i++) { prev[i] = cur[i]; cur[i] = $i } }
    END {
        if (!n) { print "error: no rows for " model " in the results" > "/dev/stderr"; exit 1 }
        printf "---\nlayout: post\ntitle: \"%s\"\ncommit: %s\ntok_s: %s\nsummary: \"\"\n# walkthrough: /walkthroughs/NAME/   (a page in docs/walkthroughs/NAME/index.html)\n---\n\n", title, cur[2], cur[10]
        print "## What changed\n\nTODO\n\n## Numbers\n"
        if (n > 1) print "| | Before | After | Change |\n|---|---|---|---|"
        else print "| | Measured |\n|---|---|"
        line("Generate (tokens/s)", 10, ratio(prev[10], cur[10]))
        line("Prompt (tokens/s)", 8, ratio(prev[8], cur[8]))
        line("Time to first token (s)", 9, ratio(prev[9], cur[9]))
        line("Load (s)", 7, ratio(prev[7], cur[7]))
        line("Peak memory (GB)", 11, ratio(prev[11], cur[11]))
        line("Swap growth (MB)", 12, "")
        line("Logits difference from the original", 13, "")
        line("Reply identical to the original", 14, "")
        printf "\nQwen3-8B, %s threads, commit `%s`", cur[4], cur[2]
        if (n > 1) printf " (before: `%s`)", prev[2]
        print ". Median of three runs of 64 tokens.\n\n## Why it worked\n\nTODO\n\n## Next\n\nTODO"
    }' "$CSV" > "$post" || { rm -f "$post"; exit 1; }

echo "wrote $post"
