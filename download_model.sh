#!/bin/sh
# Download a Qwen3 model from Hugging Face into ./model
#
#   ./download_model.sh                      Qwen/Qwen3-0.6B (about 1.5 GB)
#   ./download_model.sh Qwen/Qwen3-1.7B      a larger dense Qwen3
#   (models stored as several weight files, such as Qwen3-8B, need ./run_8b.sh)
#   ./download_model.sh Qwen/Qwen3-0.6B dir  into another directory
#
# Only three files are needed by the engine: config.json, tokenizer.json and
# model.safetensors. tokenizer_config.json is fetched too because it holds the
# official chat template, which tools/compare_chat_template.py checks against.
set -e

REPO="${1:-Qwen/Qwen3-0.6B}"
DIR="${2:-model}"
BASE="${HF_ENDPOINT:-https://huggingface.co}/$REPO/resolve/main"

# Catch a mistyped repo name before downloading anything (it must look like owner/name).
case "$REPO" in
    */*) ;;
    *) echo "error: '$REPO' is not a Hugging Face repo name; expected owner/name, e.g. Qwen/Qwen3-0.6B" >&2; exit 2 ;;
esac
case "$REPO" in
    *[!A-Za-z0-9._/-]*) echo "error: '$REPO' contains characters a repo name cannot have" >&2; exit 2 ;;
esac

mkdir -p "$DIR"
for f in config.json tokenizer.json tokenizer_config.json model.safetensors; do
    if [ -f "$DIR/$f" ]; then
        echo "== $f (already downloaded)"
        continue
    fi
    echo "== $f"
    # Download to NAME.part and rename when complete, so an interrupted run
    # resumes and a finished file is skipped on the next run.
    # -L follows redirects, -C - resumes a partial file, --fail stops on HTTP errors
    curl -L --fail --progress-bar -C - -o "$DIR/$f.part" "$BASE/$f"
    mv "$DIR/$f.part" "$DIR/$f"
done

# A wrong URL can still return a web page; make sure config.json really is JSON.
case "$(head -c 1 "$DIR/config.json")" in
    "{") ;;
    *) echo "error: $DIR/config.json is not JSON; the download did not return model files" >&2; exit 1 ;;
esac

echo
echo "Done. Files in $DIR/:"
ls -lh "$DIR"
echo
echo "Next:  ./build.sh test && ./qwen3 -m $DIR -p \"Why is the sky blue?\""
