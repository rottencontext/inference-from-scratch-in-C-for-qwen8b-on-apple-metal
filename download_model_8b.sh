#!/bin/sh
# Download Qwen3-8B from Hugging Face into ./model-8b
#
#   ./download_model_8b.sh          into ./model-8b (about 16.4 GB)
#   ./download_model_8b.sh dir      into another directory
#
# Unlike Qwen3-0.6B, which is one model.safetensors, the 8B checkpoint is split
# into several model-0000N-of-0000M.safetensors shards. The shard names are read
# from model.safetensors.index.json, which the engine also uses to find them.
#
# Running it again is safe: finished files are skipped and a partial file
# resumes where it stopped.
#
# The engine widens the weights to float32, so running the model takes about
# 33 GB of RAM plus 0.3 MB per token of context (roughly 34 GB at the default
# 4096-token window).
set -e

REPO="Qwen/Qwen3-8B"
DIR="${1:-model-8b}"
BASE="https://huggingface.co/$REPO/resolve/main"
NEED_RAM_GB=36

# Size in bytes the server reports for a file (empty if it does not say).
remote_size() {
    curl -sIL --fail "$BASE/$1" | tr -d '\r' |
        awk 'tolower($1) == "content-length:" { n = $2 } END { print n }'
}

local_size() {
    if [ -f "$1" ]; then wc -c < "$1" | tr -d ' '; else echo 0; fi
}

fetch() {
    f="$1"
    want="$(remote_size "$f")"
    have="$(local_size "$DIR/$f")"
    if [ -n "$want" ] && [ "$have" = "$want" ]; then
        echo "== $f (already complete)"
        return
    fi
    # A local file larger than the remote one cannot be resumed; start it over.
    if [ -n "$want" ] && [ "$have" -gt "$want" ]; then rm -f "$DIR/$f"; fi
    echo "== $f"
    # -L follows redirects, -C - resumes a partial download, --fail stops on HTTP errors
    curl -L --fail --progress-bar -C - -o "$DIR/$f" "$BASE/$f"
    have="$(local_size "$DIR/$f")"
    if [ -n "$want" ] && [ "$have" != "$want" ]; then
        echo "error: $DIR/$f is $have bytes, expected $want; run this script again to resume" >&2
        exit 1
    fi
}

# Warn early if this machine is unlikely to hold the model in memory.
case "$(uname -s)" in
    Darwin) ram_kb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 )) ;;
    Linux)  ram_kb="$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo 2>/dev/null)" ;;
    *)      ram_kb="" ;;
esac
ram_gb=$(( ${ram_kb:-0} / 1048576 ))
if [ "$ram_gb" -gt 0 ] && [ "$ram_gb" -lt "$NEED_RAM_GB" ]; then
    echo "warning: this machine has $ram_gb GB of RAM; Qwen3-8B needs about 34 GB and will swap heavily" >&2
fi

mkdir -p "$DIR"

# The small files first, including the index that lists the weight shards.
for f in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
    fetch "$f"
done

# A wrong URL can still return a web page; make sure config.json really is JSON.
case "$(head -c 1 "$DIR/config.json")" in
    "{") ;;
    *) echo "error: $DIR/config.json is not JSON; the download did not return model files" >&2; exit 1 ;;
esac

# The index maps every tensor name to the shard that holds it.
SHARDS="$(grep -o 'model-[0-9]*-of-[0-9]*\.safetensors' "$DIR/model.safetensors.index.json" | sort -u)"
if [ -z "$SHARDS" ]; then
    echo "error: no weight shards listed in $DIR/model.safetensors.index.json" >&2
    exit 1
fi
echo "weight shards: $(echo "$SHARDS" | wc -l | tr -d ' ')"

for f in $SHARDS; do
    fetch "$f"
done

echo
echo "Done. Files in $DIR/:"
ls -lh "$DIR"
echo
echo "Next:  ./build.sh && ./qwen3 -m $DIR -p \"Why is the sky blue?\""
