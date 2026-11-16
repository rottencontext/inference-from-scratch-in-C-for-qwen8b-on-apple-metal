#!/bin/sh
# Build and test the Qwen3 inference engine.
#
#   ./build.sh           optimized build -> ./qwen3
#   ./build.sh test      build, then self-test: forward pass against reference
#                        logits, plus the tokenizer once model/ is downloaded
#   ./build.sh debug     slow build with address/undefined-behaviour checks -> ./qwen3-debug
#   ./build.sh tables    regenerate unicode_tables.h (needs python3; rarely necessary)
#   ./build.sh clean
#
# Override the compiler or flags with CC=... or CFLAGS=...
set -e
cd "$(dirname "$0")"

CC="${CC:-cc}"
WARN="-std=c11 -Wall -Wextra -Wno-psabi"
LIBS="-lm -lpthread"

# Let the compiler use every instruction this machine has (NEON on Apple
# Silicon, AVX on x86). The flag is spelled differently per architecture.
case "$(uname -m)" in
    x86_64)        ARCH="-march=native" ;;
    arm64|aarch64) ARCH="-mcpu=native" ;;
    *)             ARCH="" ;;
esac

build() {
    if [ -n "${CFLAGS:-}" ]; then
        echo "$CC $CFLAGS $WARN -o qwen3 qwen3.c $LIBS"
        $CC $CFLAGS $WARN -o qwen3 qwen3.c $LIBS
    elif $CC -O3 $ARCH $WARN -o qwen3 qwen3.c $LIBS 2>/dev/null; then
        echo "built ./qwen3  ($CC -O3 $ARCH)"
    else
        # some older compilers do not know the native-CPU flag
        echo "$CC -O3 $WARN -o qwen3 qwen3.c $LIBS"
        $CC -O3 $WARN -o qwen3 qwen3.c $LIBS
    fi
}

run_tests() {
    $CC -O2 -o tests/cmp_f32 tests/cmp_f32.c -lm
    ids="$(cat tests/tiny/ids.txt)"

    # 1. Forward pass: a 0.4M-parameter random model with the Qwen3 architecture,
    #    300 tokens, compared with logits computed by Hugging Face transformers.
    echo "forward pass (1 thread):"
    ./qwen3 -m tests/tiny -c 512 --threads 1 --dump-logits "$ids" tests/out.bin 2>/dev/null
    ./tests/cmp_f32 tests/out.bin tests/tiny/ref_logits.bin 1e-3
    echo "forward pass (default threads):"
    ./qwen3 -m tests/tiny -c 512 --dump-logits "$ids" tests/out.bin 2>/dev/null
    ./tests/cmp_f32 tests/out.bin tests/tiny/ref_logits.bin 1e-3

    # 2. Tokenizer: 489 strings compared with ids from Hugging Face tokenizers.
    #    Needs the real tokenizer.json, so it is skipped until you download it.
    echo "tokenizer:"
    if [ -f model/tokenizer.json ]; then
        ./qwen3 -m model --encode-json tests/tokenizer_cases.json > tests/out.txt
        if cmp -s tests/out.txt tests/tokenizer_expected.txt; then
            echo "ok: 489 strings match"
        else
            echo "FAIL: see 'diff tests/out.txt tests/tokenizer_expected.txt'"
            exit 1
        fi
    else
        echo "skipped (run ./download_model.sh first)"
    fi
}

case "${1:-build}" in
    build)  build ;;
    test)   build; run_tests ;;
    debug)  $CC -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer $WARN -o qwen3-debug qwen3.c $LIBS
            echo "built ./qwen3-debug" ;;
    tables) python3 tools/gen_unicode_tables.py > unicode_tables.h ;;
    clean)  rm -f qwen3 qwen3-debug tests/cmp_f32 tests/out.bin tests/out.txt ;;
    *)      echo "usage: $0 [build|test|debug|tables|clean]" >&2; exit 2 ;;
esac
