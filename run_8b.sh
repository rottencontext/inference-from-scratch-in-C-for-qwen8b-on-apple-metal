#!/bin/sh
# Download Qwen3-8B into ./model-8b (once) and run it with the engine.
#
#   ./run_8b.sh                        interactive chat
#   ./run_8b.sh -p "Why is the sky blue?"
#   ./run_8b.sh --no-think -c 8192     every ./qwen3 option is passed through
#   ./run_8b.sh --download-only        fetch the files and stop
#
# What to expect from the 8B model on this engine:
#   Download   about 16.4 GB: five weight files plus the tokenizer and config.
#   Memory     about 34 GB. The engine widens the 16-bit weights to float32,
#              so 8.2 billion parameters take 32.8 GB, plus the context cache.
#   Speed      slow. Every token reads all 32.8 GB of weights once.
#
# Do not put a "# comment" after the command when you type it in zsh: at an
# interactive prompt zsh passes the words of the comment on as arguments.
set -e
cd "$(dirname "$0")"

REPO="Qwen/Qwen3-8B"
DIR="model-8b"
BASE="${HF_ENDPOINT:-https://huggingface.co}/$REPO/resolve/main"
NEED_RAM_GB=40    # 34 GB for the model, the rest for macOS and everything else
NEED_DISK_GB=18

DOWNLOAD_ONLY=0
if [ "${1:-}" = "--download-only" ]; then
    DOWNLOAD_ONLY=1
    shift
fi

# ---- 1. Is there enough memory? Refuse rather than push the machine into swap.
if [ "$(uname -s)" = "Darwin" ]; then
    ram_bytes="$(sysctl -n hw.memsize)"
else
    ram_bytes="$(awk '/MemTotal/ { printf "%.0f", $2 * 1024 }' /proc/meminfo)"
fi
ram_gb=$((ram_bytes / 1073741824))
if [ "$ram_gb" -lt "$NEED_RAM_GB" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "error: this machine has $ram_gb GB of memory; Qwen3-8B needs about 34 GB on this engine." >&2
    echo "       Use ./download_model.sh Qwen/Qwen3-4B model-4b for a smaller model, or set FORCE=1 to try anyway." >&2
    exit 1
fi

# ---- 2. Download. Each file goes to NAME.part first and is renamed when
#         complete, so an interrupted run resumes and a finished file is skipped.
fetch() {
    if [ -f "$DIR/$1" ]; then
        echo "== $1 (already downloaded)"
        return 0
    fi
    echo "== $1"
    # -L follows redirects, -C - resumes a partial file, --fail stops on HTTP errors
    curl -L --fail --progress-bar -C - -o "$DIR/$1.part" "$BASE/$1"
    mv "$DIR/$1.part" "$DIR/$1"
}

mkdir -p "$DIR"
if [ ! -f "$DIR/model.safetensors.index.json" ]; then
    free_gb=$(($(df -k "$DIR" | awk 'NR == 2 { print $4 }') / 1048576))
    if [ "$free_gb" -lt "$NEED_DISK_GB" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "error: only $free_gb GB free on this disk; the download needs about 16.4 GB." >&2
        exit 1
    fi
fi

for f in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
    fetch "$f"
done

# A wrong URL can still return a web page; make sure config.json really is JSON.
case "$(head -c 1 "$DIR/config.json")" in
    "{") ;;
    *) echo "error: $DIR/config.json is not JSON; the download did not return model files" >&2; exit 1 ;;
esac

# The index lists which weight file holds each tensor; take the file names from it.
shards="$(grep -o 'model-[0-9]*-of-[0-9]*\.safetensors' "$DIR/model.safetensors.index.json" | sort -u)"
if [ -z "$shards" ]; then
    echo "error: no weight files are listed in $DIR/model.safetensors.index.json" >&2
    exit 1
fi
for f in $shards; do
    fetch "$f"
done

# The index also records how many bytes of weights there are in total. The
# files must add up to at least that, or something was cut short.
want_bytes="$(grep -o '"total_size": *[0-9]*' "$DIR/model.safetensors.index.json" | grep -o '[0-9]*$' || true)"
have_bytes=0
for f in $shards; do
    have_bytes=$((have_bytes + $(wc -c < "$DIR/$f")))
done
if [ -n "$want_bytes" ] && [ "$have_bytes" -lt "$want_bytes" ]; then
    echo "error: the weight files hold $have_bytes bytes but the index expects $want_bytes." >&2
    echo "       Delete the smallest-looking model-*.safetensors in $DIR/ and run this again." >&2
    exit 1
fi
echo "model files ready in $DIR/ ($((have_bytes / 1073741824)) GB of weights)"

if [ "$DOWNLOAD_ONLY" = "1" ]; then
    exit 0
fi

# ---- 3. Build the engine if it is missing or older than its source.
if [ ! -x ./qwen3 ] || [ qwen3.c -nt ./qwen3 ]; then
    ./build.sh
fi

# ---- 4. Run. Loading converts 8.2 billion numbers, so give it a minute.
echo "loading Qwen3-8B (about 34 GB of memory; the first load reads 16 GB from disk)..."
exec ./qwen3 -m "$DIR" "$@"
