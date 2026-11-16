/*
 * qwen3.c -- a complete inference engine for Qwen3 language models in one C file.
 *
 * No Python, no ML framework, no BLAS. It reads the three files Hugging Face
 * ships for a model and runs it:
 *
 *     config.json        the architecture hyper-parameters
 *     tokenizer.json     the vocabulary and BPE merge rules
 *     model.safetensors  the weights
 *
 * The file is organised in the order data flows through an LLM:
 *
 *     1. Utilities            memory, files, timing
 *     2. JSON parser          all three input files start with JSON
 *     3. Unicode              UTF-8, character classes, NFC normalization
 *     4. Tokenizer            text  -> token ids  (byte-level BPE)
 *     5. Safetensors loader   file  -> float arrays
 *     6. Thread pool + math   the dot product everything else is built on
 *     7. The model            token ids -> logits (the transformer itself)
 *     8. Sampler              logits -> next token id
 *     9. Chat loop + CLI      the prompt template and the generation loop
 *
 * Build:  ./build.sh      (or: cc -O3 -o qwen3 qwen3.c -lm -lpthread)
 * Run:    ./qwen3 -m model -p "Why is the sky blue?"
 *
 * Written for Qwen3-0.6B but nothing is hard-coded to that size: every
 * dimension comes from config.json, so the larger dense Qwen3 models work too
 * (the MoE ones do not; they have a different layer type).
 */

#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#else
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "unicode_tables.h"

/* ========================================================================== *
 * 1. Utilities
 * ========================================================================== */

_Noreturn static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory (wanted %zu bytes)\n", n);
    return p;
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) die("out of memory (wanted %zu x %zu bytes)\n", n, size);
    return p;
}

static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory (wanted %zu bytes)\n", n);
    return p;
}

/* Float arrays are aligned to 64 bytes so SIMD loads never straddle a cache line. */
static float *alloc_floats(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, (n ? n : 1) * sizeof(float)) != 0)
        die("out of memory (wanted %zu floats)\n", n);
    return (float *)p;
}

/* Read a whole file into memory. The buffer is NUL-terminated (the JSON parser relies on it). */
static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s: %s\n", path, strerror(errno));
    if (fseek(f, 0, SEEK_END) != 0) die("cannot seek in %s\n", path);
    long size = ftell(f);
    if (size < 0) die("cannot get the size of %s\n", path);
    rewind(f);
    char *buf = xmalloc((size_t)size + 1);
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) die("cannot read %s\n", path);
    fclose(f);
    buf[size] = 0;
    if (len_out) *len_out = (size_t)size;
    return buf;
}

static char *path_join(const char *dir, const char *name) {
    size_t a = strlen(dir), b = strlen(name);
    char *p = xmalloc(a + b + 2);
    memcpy(p, dir, a);
    p[a] = '/';
    memcpy(p + a + 1, name, b + 1);
    return p;
}

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* A growable array of ints: token id lists are built with this everywhere. */
typedef struct {
    int *v;
    int n, cap;
} IntVec;

static void iv_push(IntVec *a, int x) {
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 64;
        a->v = xrealloc(a->v, (size_t)a->cap * sizeof(int));
    }
    a->v[a->n++] = x;
}

static void iv_append(IntVec *a, const int *v, int n) {
    for (int i = 0; i < n; i++) iv_push(a, v[i]);
}

/* ========================================================================== *
 * 2. JSON parser
 *
 * A small recursive-descent parser that builds a tree. It is used for
 * config.json (tiny), the safetensors header (a few hundred entries) and
 * tokenizer.json (about 11 MB and 300k strings), so it allocates from an
 * arena: one bump pointer, freed all at once.
 * ========================================================================== */

typedef struct ArenaBlock {
    struct ArenaBlock *next;
    size_t used, cap;
} ArenaBlock;

typedef struct {
    ArenaBlock *head;
} Arena;

static void *arena_alloc(Arena *a, size_t n) {
    n = (n + 15) & ~(size_t)15;
    if (!a->head || a->head->used + n > a->head->cap) {
        size_t cap = n > ((size_t)1 << 20) ? n : ((size_t)1 << 20);
        ArenaBlock *b = xmalloc(sizeof(ArenaBlock) + cap);
        b->next = a->head;
        b->used = 0;
        b->cap = cap;
        a->head = b;
    }
    void *p = (char *)(a->head + 1) + a->head->used;
    a->head->used += n;
    return p;
}

static void arena_free(Arena *a) {
    while (a->head) {
        ArenaBlock *next = a->head->next;
        free(a->head);
        a->head = next;
    }
}

typedef enum { J_NULL, J_FALSE, J_TRUE, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct JVal JVal;
struct JVal {
    JType type;
    double num;        /* J_NUM */
    const char *str;   /* J_STR: unescaped UTF-8 bytes, NUL-terminated */
    size_t len;        /* J_STR: length in bytes */
    JVal *items;       /* J_ARR: the elements.  J_OBJ: the values */
    JVal *keys;        /* J_OBJ: the keys (each one a J_STR), parallel to items */
    size_t count;      /* J_ARR / J_OBJ: number of elements */
};

typedef struct {
    const char *start, *p, *end;
    Arena *arena;
} JParser;

_Noreturn static void j_fail(const JParser *ps, const char *msg) {
    die("JSON: %s at byte %zu\n", msg, (size_t)(ps->p - ps->start));
}

static void j_skip_ws(JParser *ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\n' || *ps->p == '\t' || *ps->p == '\r')) ps->p++;
}

static int utf8_encode(uint32_t cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static uint32_t j_hex4(const JParser *ps, const char *s) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else j_fail(ps, "bad \\u escape");
    }
    return v;
}

/* Parse a string. ps->p points at the opening quote. Escapes are decoded to UTF-8. */
static void j_string(JParser *ps, JVal *out) {
    memset(out, 0, sizeof *out);
    const char *s = ++ps->p;
    const char *q = s; /* pass 1: find the closing quote */
    while (q < ps->end && *q != '"') q += (*q == '\\') ? 2 : 1;
    if (q >= ps->end) j_fail(ps, "unterminated string");

    /* pass 2: unescape. The decoded form is never longer than the escaped one. */
    char *buf = arena_alloc(ps->arena, (size_t)(q - s) + 1), *w = buf;
    while (s < q) {
        char c = *s++;
        if (c != '\\') {
            *w++ = c;
            continue;
        }
        c = *s++;
        switch (c) {
        case '"': *w++ = '"'; break;
        case '\\': *w++ = '\\'; break;
        case '/': *w++ = '/'; break;
        case 'b': *w++ = '\b'; break;
        case 'f': *w++ = '\f'; break;
        case 'n': *w++ = '\n'; break;
        case 'r': *w++ = '\r'; break;
        case 't': *w++ = '\t'; break;
        case 'u': {
            if (q - s < 4) j_fail(ps, "truncated \\u escape");
            uint32_t cp = j_hex4(ps, s);
            s += 4;
            /* Characters outside the BMP are written as a surrogate pair: 😀 */
            if (cp >= 0xD800 && cp <= 0xDBFF && q - s >= 6 && s[0] == '\\' && s[1] == 'u') {
                uint32_t lo = j_hex4(ps, s + 2);
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    s += 6;
                }
            }
            w += utf8_encode(cp, w);
            break;
        }
        default: j_fail(ps, "bad escape in string");
        }
    }
    *w = 0;
    out->type = J_STR;
    out->str = buf;
    out->len = (size_t)(w - buf);
    ps->p = q + 1;
}

static void j_value(JParser *ps, JVal *out, int depth) {
    memset(out, 0, sizeof *out);
    if (depth > 64) j_fail(ps, "nesting too deep");
    j_skip_ws(ps);
    if (ps->p >= ps->end) j_fail(ps, "unexpected end of input");
    char c = *ps->p;

    if (c == '"') {
        j_string(ps, out);
        return;
    }

    if (c == '{' || c == '[') {
        int is_obj = (c == '{');
        char close = is_obj ? '}' : ']';
        /* The element count is unknown up front, so collect into growable
         * heap vectors and copy into the arena once the closing bracket is seen. */
        size_t n = 0, cap = 0;
        JVal *vals = NULL, *keys = NULL;
        ps->p++;
        j_skip_ws(ps);
        if (ps->p < ps->end && *ps->p == close) {
            ps->p++;
        } else {
            for (;;) {
                if (n == cap) {
                    cap = cap ? cap * 2 : 8;
                    vals = xrealloc(vals, cap * sizeof *vals);
                    if (is_obj) keys = xrealloc(keys, cap * sizeof *keys);
                }
                if (is_obj) {
                    j_skip_ws(ps);
                    if (ps->p >= ps->end || *ps->p != '"') j_fail(ps, "expected an object key");
                    j_string(ps, &keys[n]);
                    j_skip_ws(ps);
                    if (ps->p >= ps->end || *ps->p != ':') j_fail(ps, "expected ':'");
                    ps->p++;
                }
                j_value(ps, &vals[n], depth + 1);
                n++;
                j_skip_ws(ps);
                if (ps->p >= ps->end) j_fail(ps, "unexpected end of input");
                if (*ps->p == ',') {
                    ps->p++;
                    continue;
                }
                if (*ps->p == close) {
                    ps->p++;
                    break;
                }
                j_fail(ps, "expected ',' or a closing bracket");
            }
        }
        out->type = is_obj ? J_OBJ : J_ARR;
        out->count = n;
        out->items = arena_alloc(ps->arena, n * sizeof(JVal));
        if (n) memcpy(out->items, vals, n * sizeof(JVal));
        free(vals);
        if (is_obj) {
            out->keys = arena_alloc(ps->arena, n * sizeof(JVal));
            if (n) memcpy(out->keys, keys, n * sizeof(JVal));
            free(keys);
        }
        return;
    }

    size_t left = (size_t)(ps->end - ps->p);
    if (left >= 4 && !memcmp(ps->p, "true", 4)) {
        out->type = J_TRUE;
        ps->p += 4;
        return;
    }
    if (left >= 5 && !memcmp(ps->p, "false", 5)) {
        out->type = J_FALSE;
        ps->p += 5;
        return;
    }
    if (left >= 4 && !memcmp(ps->p, "null", 4)) {
        out->type = J_NULL;
        ps->p += 4;
        return;
    }

    char *num_end;
    double v = strtod(ps->p, &num_end); /* safe: the input is NUL-terminated */
    if (num_end == ps->p) j_fail(ps, "unexpected character");
    out->type = J_NUM;
    out->num = v;
    ps->p = num_end;
}

/* Parse `text` (which must have a NUL at text[len]) into a tree allocated from `arena`. */
static void json_parse(const char *text, size_t len, Arena *arena, JVal *root) {
    JParser ps = {text, text, text + len, arena};
    j_value(&ps, root, 0);
    j_skip_ws(&ps);
    if (ps.p != ps.end) j_fail(&ps, "trailing data after the JSON value");
}

/* Look up a key in an object (linear scan: fine for the small objects we query). */
static const JVal *j_get(const JVal *obj, const char *key) {
    if (!obj || obj->type != J_OBJ) return NULL;
    size_t klen = strlen(key);
    for (size_t i = 0; i < obj->count; i++)
        if (obj->keys[i].len == klen && !memcmp(obj->keys[i].str, key, klen)) return &obj->items[i];
    return NULL;
}

static double j_num(const JVal *obj, const char *key, double dflt) {
    const JVal *v = j_get(obj, key);
    return (v && v->type == J_NUM) ? v->num : dflt;
}

static int j_str_is(const JVal *v, const char *s) {
    return v && v->type == J_STR && v->len == strlen(s) && !memcmp(v->str, s, v->len);
}

/* ========================================================================== *
 * 3. Unicode
 *
 * The tokenizer works on bytes, but two of its steps are defined on
 * characters: NFC normalization, and a regex that splits text into "words"
 * using the classes letter / number / whitespace.
 * ========================================================================== */

/* Decode one UTF-8 character from s[0..n). Returns the number of bytes used.
 * A malformed byte is consumed alone and reported as U+FFFD. */
static int utf8_decode(const unsigned char *s, size_t n, uint32_t *cp) {
    unsigned char c = s[0];
    int len;
    uint32_t v;
    if (c < 0x80) {
        *cp = c;
        return 1;
    } else if (c >= 0xC2 && c <= 0xDF) {
        len = 2;
        v = c & 0x1F;
    } else if (c >= 0xE0 && c <= 0xEF) {
        len = 3;
        v = c & 0x0F;
    } else if (c >= 0xF0 && c <= 0xF4) {
        len = 4;
        v = c & 0x07;
    } else {
        *cp = 0xFFFD;
        return 1;
    }
    if ((size_t)len > n) {
        *cp = 0xFFFD;
        return 1;
    }
    for (int i = 1; i < len; i++) {
        if ((s[i] & 0xC0) != 0x80) {
            *cp = 0xFFFD;
            return 1;
        }
        v = (v << 6) | (s[i] & 0x3F);
    }
    if ((len == 3 && v < 0x800) || (len == 4 && (v < 0x10000 || v > 0x10FFFF)) || (v >= 0xD800 && v <= 0xDFFF)) {
        *cp = 0xFFFD;
        return 1;
    }
    *cp = v;
    return len;
}

/* Binary search in a sorted table of inclusive [lo, hi] ranges. */
static int in_ranges(const uint32_t (*r)[2], int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid - 1;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}

static int uni_is_letter(uint32_t c) { /* regex \p{L} */
    if (c < 0x80) return ((c | 0x20) >= 'a' && (c | 0x20) <= 'z');
    return in_ranges(UNI_LETTER, UNI_LETTER_N, c);
}

static int uni_is_number(uint32_t c) { /* regex \p{N} */
    if (c < 0x80) return (c >= '0' && c <= '9');
    return in_ranges(UNI_NUMBER, UNI_NUMBER_N, c);
}

static int uni_is_space(uint32_t c) { /* regex \s : the Unicode White_Space property */
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

/* --- NFC normalization ----------------------------------------------------
 *
 * The same visible character can be stored in more than one way: "é" is
 * either U+00E9, or "e" followed by the combining accent U+0301. NFC picks
 * the composed spelling. The tokenizer was trained on NFC text, so input has
 * to be normalized the same way or identical-looking text would produce
 * different tokens.
 *
 * The algorithm (Unicode Standard Annex #15) has three steps:
 *   1. decompose every character fully,
 *   2. sort each run of combining marks by combining class,
 *   3. recompose pairs wherever a composite exists.
 * Hangul syllables are handled by arithmetic instead of tables.
 */

static int uni_ccc(uint32_t c) { /* canonical combining class; 0 for ordinary characters */
    if (c < 0x300) return 0;
    int lo = 0, hi = UNI_CCC_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (c < UNI_CCC[mid][0]) hi = mid - 1;
        else if (c > UNI_CCC[mid][1]) lo = mid + 1;
        else return (int)UNI_CCC[mid][2];
    }
    return 0;
}

static const uint32_t *uni_decomp(uint32_t c) { /* row {cp, len, d0..d3}, or NULL */
    if (c < 0xC0) return NULL;
    int lo = 0, hi = UNI_DECOMP_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (c < UNI_DECOMP[mid][0]) hi = mid - 1;
        else if (c > UNI_DECOMP[mid][0]) lo = mid + 1;
        else return UNI_DECOMP[mid];
    }
    return NULL;
}

static uint32_t uni_compose(uint32_t a, uint32_t b) { /* composite of the pair, or 0 */
    int lo = 0, hi = UNI_COMPOSE_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        const uint32_t *r = UNI_COMPOSE[mid];
        if (a < r[0] || (a == r[0] && b < r[1])) hi = mid - 1;
        else if (a > r[0] || (a == r[0] && b > r[1])) lo = mid + 1;
        else return r[2];
    }
    return 0;
}

enum { /* Hangul syllable arithmetic, straight from the Unicode standard */
    H_SBASE = 0xAC00, H_LBASE = 0x1100, H_VBASE = 0x1161, H_TBASE = 0x11A7,
    H_LCOUNT = 19, H_VCOUNT = 21, H_TCOUNT = 28, H_SCOUNT = H_LCOUNT * H_VCOUNT * H_TCOUNT
};

/* Return a newly allocated NFC-normalized copy of the UTF-8 string s[0..n). */
static unsigned char *nfc_normalize(const unsigned char *s, size_t n, size_t *out_n) {
    /* Fast path. Every character below U+0300 is already in NFC form and no
     * combining mark lives there, so text made only of those needs no work.
     * In UTF-8 those characters are exactly the ones with no byte >= 0xCC. */
    int needs_work = 0;
    for (size_t i = 0; i < n; i++)
        if (s[i] >= 0xCC) {
            needs_work = 1;
            break;
        }
    if (!needs_work) {
        unsigned char *copy = xmalloc(n + 1);
        memcpy(copy, s, n);
        copy[n] = 0;
        *out_n = n;
        return copy;
    }

    /* Step 1: decode and decompose. A character decomposes into at most 4, and
     * only characters of 2+ bytes decompose, so 3n is a safe bound. */
    size_t cap = n * 3 + 16, m = 0;
    uint32_t *buf = xmalloc(cap * sizeof *buf);
    uint8_t *cc = xmalloc(cap);
    for (size_t i = 0; i < n;) {
        uint32_t c;
        i += (size_t)utf8_decode(s + i, n - i, &c);
        if (c >= H_SBASE && c < H_SBASE + H_SCOUNT) {
            uint32_t si = c - H_SBASE;
            buf[m++] = H_LBASE + si / (H_VCOUNT * H_TCOUNT);
            buf[m++] = H_VBASE + (si % (H_VCOUNT * H_TCOUNT)) / H_TCOUNT;
            if (si % H_TCOUNT) buf[m++] = H_TBASE + si % H_TCOUNT;
            continue;
        }
        const uint32_t *d = uni_decomp(c);
        if (d) {
            for (uint32_t k = 0; k < d[1]; k++) buf[m++] = d[2 + k];
        } else {
            buf[m++] = c;
        }
    }

    /* Step 2: canonical ordering. A stable insertion sort that only moves
     * combining marks (class != 0) and never moves one past a class-0 character. */
    for (size_t i = 0; i < m; i++) cc[i] = (uint8_t)uni_ccc(buf[i]);
    for (size_t i = 1; i < m; i++) {
        if (cc[i] == 0) continue;
        uint32_t c = buf[i];
        uint8_t k = cc[i];
        size_t j = i;
        while (j > 0 && cc[j - 1] > k) {
            buf[j] = buf[j - 1];
            cc[j] = cc[j - 1];
            j--;
        }
        buf[j] = c;
        cc[j] = k;
    }

    /* Step 3: compose. `starter` is the last class-0 character written; a later
     * character may merge into it unless a mark of equal or higher class sits
     * in between (max_cc tracks the highest class seen since the starter). */
    size_t w = 0;
    long starter = -1;
    int max_cc = -1;
    for (size_t r = 0; r < m; r++) {
        uint32_t c = buf[r];
        int k = cc[r];
        if (starter >= 0 && k > max_cc) {
            uint32_t st = buf[starter], comp;
            if (st >= H_LBASE && st < H_LBASE + H_LCOUNT && c >= H_VBASE && c < H_VBASE + H_VCOUNT)
                comp = H_SBASE + ((st - H_LBASE) * H_VCOUNT + (c - H_VBASE)) * H_TCOUNT;
            else if (st >= H_SBASE && st < H_SBASE + H_SCOUNT && (st - H_SBASE) % H_TCOUNT == 0 && c > H_TBASE &&
                     c < H_TBASE + H_TCOUNT)
                comp = st + (c - H_TBASE);
            else
                comp = uni_compose(st, c);
            if (comp) {
                buf[starter] = comp;
                continue;
            }
        }
        buf[w] = c;
        if (k) {
            if (k > max_cc) max_cc = k;
        } else {
            starter = (long)w;
            max_cc = -1;
        }
        w++;
    }

    unsigned char *out = xmalloc(w * 4 + 1);
    size_t o = 0;
    for (size_t i = 0; i < w; i++) o += (size_t)utf8_encode(buf[i], (char *)out + o);
    out[o] = 0;
    free(buf);
    free(cc);
    *out_n = o;
    return out;
}

/* ========================================================================== *
 * 4. Tokenizer
 *
 * Qwen uses byte-level BPE, the same scheme as GPT-2:
 *
 *   - The base alphabet is the 256 possible bytes, so any input can be encoded.
 *   - `merges` is an ordered list of pairs. "Merge the pair (a, b) into the
 *     token ab." Earlier in the list means higher priority.
 *   - To encode a word: start with its bytes, then repeatedly apply the
 *     highest-priority merge that is present, until none applies.
 *
 * Before BPE runs, the text is NFC-normalized and split into word-like
 * pieces by a regex; merges never cross a piece boundary.
 *
 * On top of the BPE vocabulary sit the "added tokens" such as <|im_start|>.
 * They are never produced by BPE; the chat template inserts them by id.
 * ========================================================================== */

typedef struct {
    int n_ids;              /* size of the id space */
    unsigned char *pool;    /* the bytes of every token, concatenated */
    uint32_t *tok_off;      /* token id -> offset into pool */
    uint32_t *tok_len;      /* token id -> length in bytes */

    int32_t *vh;            /* hash table: token bytes -> id (BPE vocabulary only) */
    uint32_t vh_mask;

    uint64_t *mh_key;       /* hash table: (left id, right id) -> merge rank and merged id */
    int32_t *mh_rank, *mh_id;
    uint32_t mh_mask;

    int byte_tok[256];      /* the token id of each single byte */
    int *added;             /* ids of the added tokens */
    int n_added;

    int im_start, im_end, endoftext, think, think_end; /* ids the chat loop needs */
} Tokenizer;

static uint32_t hash_bytes(const unsigned char *s, size_t n) { /* FNV-1a */
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= s[i];
        h *= 16777619u;
    }
    return h;
}

static uint64_t hash_u64(uint64_t x) { /* the murmur3 finalizer */
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static uint32_t pow2_at_least(size_t n) {
    uint32_t p = 16;
    while (p < n) p <<= 1;
    return p;
}

static int tok_lookup(const Tokenizer *t, const unsigned char *s, size_t n) {
    for (uint32_t i = hash_bytes(s, n) & t->vh_mask;; i = (i + 1) & t->vh_mask) {
        int id = t->vh[i];
        if (id < 0) return -1;
        if (t->tok_len[id] == n && !memcmp(t->pool + t->tok_off[id], s, n)) return id;
    }
}

/* Is (a, b) a merge? Returns its rank (lower = earlier = higher priority) or -1. */
static int merge_lookup(const Tokenizer *t, int a, int b, int *merged) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    for (uint32_t i = (uint32_t)hash_u64(key) & t->mh_mask;; i = (i + 1) & t->mh_mask) {
        if (t->mh_rank[i] < 0) return -1;
        if (t->mh_key[i] == key) {
            *merged = t->mh_id[i];
            return t->mh_rank[i];
        }
    }
}

/* tokenizer.json stores tokens as printable text, not raw bytes. Each of the
 * 256 bytes is assigned one printable character: bytes that already are
 * printable map to themselves, and the other 68 map to U+0100..U+0143 in
 * order. (That is why a leading space shows up as "Ġ", U+0120.)
 * This builds the inverse table: character -> byte. */
static void build_bytelevel_table(int cp2byte[324]) {
    for (int i = 0; i < 324; i++) cp2byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        if (printable) cp2byte[b] = b;
        else cp2byte[256 + n++] = b;
    }
}

/* Turn one token as spelled in tokenizer.json back into its raw bytes.
 * Returns the byte count, or (size_t)-1 if a character is outside the alphabet. */
static size_t bytelevel_decode(const char *s, size_t n, const int cp2byte[324], unsigned char *out) {
    size_t o = 0;
    for (size_t i = 0; i < n;) {
        uint32_t cp;
        i += (size_t)utf8_decode((const unsigned char *)s + i, n - i, &cp);
        if (cp >= 324 || cp2byte[cp] < 0) return (size_t)-1;
        out[o++] = (unsigned char)cp2byte[cp];
    }
    return o;
}

static int tok_find_added(const Tokenizer *t, const char *content) {
    size_t n = strlen(content);
    for (int i = 0; i < t->n_added; i++) {
        int id = t->added[i];
        if (t->tok_len[id] == n && !memcmp(t->pool + t->tok_off[id], content, n)) return id;
    }
    return -1;
}

static void tok_load(Tokenizer *t, const char *path, int min_ids) {
    memset(t, 0, sizeof *t);
    size_t len;
    char *text = read_file(path, &len);
    Arena arena = {0};
    JVal root;
    json_parse(text, len, &arena, &root);

    const JVal *model = j_get(&root, "model");
    const JVal *vocab = j_get(model, "vocab");
    const JVal *merges = j_get(model, "merges");
    const JVal *added = j_get(&root, "added_tokens");
    if (!j_str_is(j_get(model, "type"), "BPE")) die("%s: only BPE tokenizers are supported\n", path);
    if (!vocab || vocab->type != J_OBJ || !merges || merges->type != J_ARR)
        die("%s: missing model.vocab or model.merges\n", path);
    size_t n_added = (added && added->type == J_ARR) ? added->count : 0;

    /* Size the id space and the byte pool. */
    int max_id = -1;
    size_t pool_cap = 1, max_tok = 1;
    for (size_t i = 0; i < vocab->count; i++) {
        if (vocab->items[i].type != J_NUM) die("%s: vocab ids must be numbers\n", path);
        int id = (int)vocab->items[i].num;
        if (id < 0) die("%s: negative token id\n", path);
        if (id > max_id) max_id = id;
        pool_cap += vocab->keys[i].len;
        if (vocab->keys[i].len > max_tok) max_tok = vocab->keys[i].len;
    }
    for (size_t i = 0; i < n_added; i++) {
        const JVal *id = j_get(&added->items[i], "id"), *content = j_get(&added->items[i], "content");
        if (!id || id->type != J_NUM || !content || content->type != J_STR) die("%s: malformed added_tokens\n", path);
        if ((int)id->num > max_id) max_id = (int)id->num;
        pool_cap += content->len;
    }
    t->n_ids = (max_id + 1 > min_ids) ? max_id + 1 : min_ids;
    t->pool = xmalloc(pool_cap);
    t->tok_off = xcalloc((size_t)t->n_ids, sizeof *t->tok_off);
    t->tok_len = xcalloc((size_t)t->n_ids, sizeof *t->tok_len);

    /* The BPE vocabulary: decode each token to bytes and index it. */
    int cp2byte[324];
    build_bytelevel_table(cp2byte);
    uint32_t vh_cap = pow2_at_least(vocab->count * 2);
    t->vh_mask = vh_cap - 1;
    t->vh = xmalloc(vh_cap * sizeof *t->vh);
    for (uint32_t i = 0; i < vh_cap; i++) t->vh[i] = -1;
    for (int b = 0; b < 256; b++) t->byte_tok[b] = -1;

    size_t pool_n = 0;
    for (size_t i = 0; i < vocab->count; i++) {
        int id = (int)vocab->items[i].num;
        unsigned char *dst = t->pool + pool_n;
        size_t n = bytelevel_decode(vocab->keys[i].str, vocab->keys[i].len, cp2byte, dst);
        if (n == (size_t)-1) die("%s: token %d is not in the byte-level alphabet\n", path, id);
        t->tok_off[id] = (uint32_t)pool_n;
        t->tok_len[id] = (uint32_t)n;
        pool_n += n;
        uint32_t h = hash_bytes(dst, n) & t->vh_mask;
        while (t->vh[h] >= 0) h = (h + 1) & t->vh_mask;
        t->vh[h] = id;
        if (n == 1) t->byte_tok[dst[0]] = id;
    }
    for (int b = 0; b < 256; b++)
        if (t->byte_tok[b] < 0) die("%s: the vocabulary has no token for byte 0x%02X\n", path, b);

    /* The merge list. Each entry is either the string "left right" or the
     * array ["left", "right"], depending on the tokenizers version that wrote it. */
    uint32_t mh_cap = pow2_at_least(merges->count * 2);
    t->mh_mask = mh_cap - 1;
    t->mh_key = xmalloc(mh_cap * sizeof *t->mh_key);
    t->mh_rank = xmalloc(mh_cap * sizeof *t->mh_rank);
    t->mh_id = xmalloc(mh_cap * sizeof *t->mh_id);
    for (uint32_t i = 0; i < mh_cap; i++) t->mh_rank[i] = -1;

    unsigned char *scratch = xmalloc(max_tok * 2 + 2);
    for (size_t i = 0; i < merges->count; i++) {
        const JVal *e = &merges->items[i];
        const char *ls, *rs;
        size_t ln, rn;
        if (e->type == J_STR) {
            const char *sp = e->len > 1 ? memchr(e->str + 1, ' ', e->len - 1) : NULL;
            if (!sp) die("%s: malformed merge #%zu\n", path, i);
            ls = e->str;
            ln = (size_t)(sp - e->str);
            rs = sp + 1;
            rn = e->len - ln - 1;
        } else if (e->type == J_ARR && e->count == 2 && e->items[0].type == J_STR && e->items[1].type == J_STR) {
            ls = e->items[0].str;
            ln = e->items[0].len;
            rs = e->items[1].str;
            rn = e->items[1].len;
        } else {
            die("%s: malformed merge #%zu\n", path, i);
        }
        if (ln > max_tok || rn > max_tok) die("%s: merge #%zu is longer than any token\n", path, i);
        size_t lb = bytelevel_decode(ls, ln, cp2byte, scratch);
        size_t rb = (lb == (size_t)-1) ? lb : bytelevel_decode(rs, rn, cp2byte, scratch + lb);
        if (lb == (size_t)-1 || rb == (size_t)-1) die("%s: merge #%zu is not in the byte-level alphabet\n", path, i);
        int a = tok_lookup(t, scratch, lb), b = tok_lookup(t, scratch + lb, rb), ab = tok_lookup(t, scratch, lb + rb);
        if (a < 0 || b < 0 || ab < 0) die("%s: merge #%zu refers to a token that is not in the vocabulary\n", path, i);
        uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
        uint32_t h = (uint32_t)hash_u64(key) & t->mh_mask;
        while (t->mh_rank[h] >= 0 && t->mh_key[h] != key) h = (h + 1) & t->mh_mask;
        if (t->mh_rank[h] < 0) { /* keep the first (highest-priority) entry if a pair repeats */
            t->mh_key[h] = key;
            t->mh_rank[h] = (int32_t)i;
            t->mh_id[h] = ab;
        }
    }
    free(scratch);

    /* Added tokens: stored as literal text, and deliberately kept out of the
     * BPE hash table so ordinary text can never turn into one. */
    t->added = xmalloc((n_added ? n_added : 1) * sizeof *t->added);
    for (size_t i = 0; i < n_added; i++) {
        int id = (int)j_get(&added->items[i], "id")->num;
        const JVal *content = j_get(&added->items[i], "content");
        memcpy(t->pool + pool_n, content->str, content->len);
        t->tok_off[id] = (uint32_t)pool_n;
        t->tok_len[id] = (uint32_t)content->len;
        pool_n += content->len;
        t->added[t->n_added++] = id;
    }

    t->im_start = tok_find_added(t, "<|im_start|>");
    t->im_end = tok_find_added(t, "<|im_end|>");
    t->endoftext = tok_find_added(t, "<|endoftext|>");
    t->think = tok_find_added(t, "<think>");
    t->think_end = tok_find_added(t, "</think>");

    arena_free(&arena);
    free(text);
}

/* --- pre-tokenization -----------------------------------------------------
 *
 * tokenizer.json splits text with this regex before BPE:
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)        1. English contractions
 *   | [^\r\n\p{L}\p{N}]?\p{L}+          2. a run of letters, with one optional leading
 *                                          non-letter (usually the space before a word)
 *   | \p{N}                             3. a single digit
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*          4. punctuation, optional leading space, trailing newlines
 *   | \s*[\r\n]+                        5. whitespace ending in a newline
 *   | \s+(?!\S)                         6. whitespace NOT followed by a visible character
 *   | \s+                               7. any other whitespace
 *
 * A regex engine tries the alternatives in order at each position and takes
 * the first that matches, backtracking inside it if needed. There are only
 * seven alternatives, so instead of a regex engine this function implements
 * each one by hand. It returns the index just past the piece that starts at i.
 */

static int is_newline(uint32_t c) { return c == '\r' || c == '\n'; }
static int is_punct(uint32_t c) { return !uni_is_space(c) && !uni_is_letter(c) && !uni_is_number(c); }

static uint32_t fold_case(uint32_t c) { /* just enough case-insensitivity for alternative 1 */
    if (c >= 'A' && c <= 'Z') return c + 32;
    if (c == 0x17F) return 's'; /* LATIN SMALL LETTER LONG S folds to s */
    return c;
}

static size_t pretok_next(const uint32_t *c, size_t n, size_t i) {
    uint32_t ch = c[i];

    /* 1. contractions: 's 't 'm 'd (two characters) and 're 've 'll (three) */
    if (ch == '\'' && i + 1 < n) {
        uint32_t a = fold_case(c[i + 1]);
        uint32_t b = (i + 2 < n) ? fold_case(c[i + 2]) : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }

    /* 2. letters. The optional prefix character is taken only when a letter
     *    follows it, which is exactly what regex backtracking would decide. */
    {
        size_t j = i;
        if (!is_newline(ch) && !uni_is_letter(ch) && !uni_is_number(ch) && i + 1 < n && uni_is_letter(c[i + 1])) j = i + 1;
        if (uni_is_letter(c[j])) {
            while (j < n && uni_is_letter(c[j])) j++;
            return j;
        }
    }

    /* 3. one digit at a time, so numbers never form long tokens */
    if (uni_is_number(ch)) return i + 1;

    /* 4. punctuation run, optionally led by one space, plus any newlines right after */
    {
        size_t j = i;
        if (ch == ' ' && i + 1 < n && is_punct(c[i + 1])) j = i + 1;
        if (is_punct(c[j])) {
            while (j < n && is_punct(c[j])) j++;
            while (j < n && is_newline(c[j])) j++;
            return j;
        }
    }

    /* 5-7. whitespace. Let [i, j) be the whole run of whitespace. */
    if (uni_is_space(ch)) {
        size_t j = i;
        while (j < n && uni_is_space(c[j])) j++;
        /* 5. the longest prefix of the run that ends in a newline */
        size_t k = j;
        while (k > i && !is_newline(c[k - 1])) k--;
        if (k > i) return k;
        /* 6. the run minus its last character, which is left to lead the next
         *    word (alternative 2 or 4). At the end of the text, the whole run. */
        if (j == n) return j;
        if (j - i >= 2) return j - 1;
        /* 7. a single whitespace character that nothing else claimed */
        return j;
    }

    return i + 1; /* not reachable: every character falls in one class above */
}

/* BPE on one piece: bytes in, token ids appended to `out`. */
static void bpe_piece(const Tokenizer *t, const unsigned char *s, size_t n, IntVec *out) {
    int stack_buf[256];
    int *sym = (n <= 256) ? stack_buf : xmalloc(n * sizeof(int));
    size_t m = n;
    for (size_t i = 0; i < n; i++) sym[i] = t->byte_tok[s[i]];

    while (m > 1) {
        /* find the adjacent pair with the best (lowest) merge rank */
        int best_rank = INT32_MAX, best_a = 0, best_b = 0, best_id = 0;
        for (size_t i = 0; i + 1 < m; i++) {
            int merged, rank = merge_lookup(t, sym[i], sym[i + 1], &merged);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_a = sym[i];
                best_b = sym[i + 1];
                best_id = merged;
            }
        }
        if (best_rank == INT32_MAX) break; /* nothing left to merge */
        /* replace every occurrence of that pair, left to right */
        size_t w = 0;
        for (size_t i = 0; i < m;) {
            if (i + 1 < m && sym[i] == best_a && sym[i + 1] == best_b) {
                sym[w++] = best_id;
                i += 2;
            } else {
                sym[w++] = sym[i++];
            }
        }
        m = w;
    }
    for (size_t i = 0; i < m; i++) iv_push(out, sym[i]);
    if (sym != stack_buf) free(sym);
}

/* Encode ordinary text: normalize, split into pieces, BPE each piece. */
static void tok_encode_text(const Tokenizer *t, const char *s, size_t n, IntVec *out) {
    if (n == 0) return;
    size_t nn;
    unsigned char *norm = nfc_normalize((const unsigned char *)s, n, &nn);
    /* decode to code points, remembering where each one starts in the byte string */
    uint32_t *cp = xmalloc((nn + 1) * sizeof *cp);
    size_t *off = xmalloc((nn + 1) * sizeof *off);
    size_t m = 0;
    for (size_t i = 0; i < nn;) {
        off[m] = i;
        i += (size_t)utf8_decode(norm + i, nn - i, &cp[m]);
        m++;
    }
    off[m] = nn;
    for (size_t i = 0; i < m;) {
        size_t j = pretok_next(cp, m, i);
        bpe_piece(t, norm + off[i], off[j] - off[i], out);
        i = j;
    }
    free(cp);
    free(off);
    free(norm);
}

/* Encode text that may contain added tokens written out literally, such as
 * "<|im_start|>". With parse_special = 0 they are treated as plain text, which
 * is what you want for anything a user typed. */
static void tok_encode(const Tokenizer *t, const char *s, size_t n, int parse_special, IntVec *out) {
    if (!parse_special) {
        tok_encode_text(t, s, n, out);
        return;
    }
    size_t seg = 0, i = 0;
    while (i < n) {
        int best = -1;
        size_t best_len = 0;
        for (int a = 0; a < t->n_added; a++) { /* longest added token starting here */
            int id = t->added[a];
            size_t len = t->tok_len[id];
            if (len > best_len && len <= n - i && !memcmp(s + i, t->pool + t->tok_off[id], len)) {
                best = id;
                best_len = len;
            }
        }
        if (best < 0) {
            i++;
            continue;
        }
        tok_encode_text(t, s + seg, i - seg, out);
        iv_push(out, best);
        i += best_len;
        seg = i;
    }
    tok_encode_text(t, s + seg, n - seg, out);
}

static void tok_encode_cstr(const Tokenizer *t, const char *s, IntVec *out) { tok_encode(t, s, strlen(s), 0, out); }

/* Decoding is just a table lookup: token id -> bytes. */
static const unsigned char *tok_bytes(const Tokenizer *t, int id, size_t *len) {
    if (id < 0 || id >= t->n_ids) {
        *len = 0;
        return t->pool;
    }
    *len = t->tok_len[id];
    return t->pool + t->tok_off[id];
}

/* ========================================================================== *
 * 5. Safetensors loader
 *
 * The file layout is:
 *
 *     8 bytes    N, a little-endian uint64
 *     N bytes    a JSON object: tensor name -> { dtype, shape, data_offsets }
 *     the rest   raw tensor data; data_offsets are relative to this point
 *
 * Qwen3 ships its weights as BF16 (bfloat16): the top 16 bits of a float32.
 * They are widened to float32 at load time so the math code stays simple.
 * ========================================================================== */

typedef struct {
    char *path;
    unsigned char *map; /* the whole file, memory-mapped */
    size_t map_len;
    const unsigned char *data; /* start of the tensor data */
    size_t data_len;
    char *hdr_text;
    Arena arena;
    JVal hdr;
} STFile;

typedef struct {
    STFile *files; /* big checkpoints are split across several files */
    int n_files;
} Weights;

static void st_open(STFile *f, const char *path) {
    memset(f, 0, sizeof *f);
    f->path = strdup(path);
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open %s: %s\n", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) die("cannot stat %s\n", path);
    f->map_len = (size_t)st.st_size;
    if (f->map_len < 8) die("%s is too small to be a safetensors file\n", path);
    f->map = mmap(NULL, f->map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (f->map == MAP_FAILED) die("cannot mmap %s: %s\n", path, strerror(errno));
    close(fd);

    uint64_t hlen = 0;
    for (int i = 7; i >= 0; i--) hlen = (hlen << 8) | f->map[i];
    if (hlen > f->map_len - 8) die("%s: corrupt header length\n", path);
    f->hdr_text = xmalloc((size_t)hlen + 1);
    memcpy(f->hdr_text, f->map + 8, (size_t)hlen);
    f->hdr_text[hlen] = 0;
    json_parse(f->hdr_text, (size_t)hlen, &f->arena, &f->hdr);
    if (f->hdr.type != J_OBJ) die("%s: header is not a JSON object\n", path);
    f->data = f->map + 8 + hlen;
    f->data_len = f->map_len - 8 - (size_t)hlen;
}

static void st_close(STFile *f) {
    munmap(f->map, f->map_len);
    arena_free(&f->arena);
    free(f->hdr_text);
    free(f->path);
}

static void weights_open(Weights *w, const char *dir) {
    memset(w, 0, sizeof *w);
    char *single = path_join(dir, "model.safetensors");
    if (file_exists(single)) {
        w->files = xmalloc(sizeof(STFile));
        st_open(&w->files[0], single);
        w->n_files = 1;
        free(single);
        return;
    }
    /* Sharded checkpoint: model.safetensors.index.json maps tensor name -> file name. */
    char *index_path = path_join(dir, "model.safetensors.index.json");
    if (!file_exists(index_path))
        die("no weights found: expected %s\n       (run ./download_model.sh to fetch them)\n", single);
    size_t len;
    char *text = read_file(index_path, &len);
    Arena arena = {0};
    JVal root;
    json_parse(text, len, &arena, &root);
    const JVal *map = j_get(&root, "weight_map");
    if (!map || map->type != J_OBJ) die("%s: no weight_map\n", index_path);
    w->files = xmalloc(map->count * sizeof(STFile) + sizeof(STFile));
    for (size_t i = 0; i < map->count; i++) {
        if (map->items[i].type != J_STR) continue;
        char *p = path_join(dir, map->items[i].str);
        int seen = 0;
        for (int k = 0; k < w->n_files; k++) seen |= !strcmp(w->files[k].path, p);
        if (!seen) st_open(&w->files[w->n_files++], p);
        free(p);
    }
    arena_free(&arena);
    free(text);
    free(index_path);
    free(single);
}

static void weights_close(Weights *w) {
    for (int i = 0; i < w->n_files; i++) st_close(&w->files[i]);
    free(w->files);
}

static const JVal *weights_find(const Weights *w, const char *name, const STFile **file) {
    for (int i = 0; i < w->n_files; i++) {
        const JVal *e = j_get(&w->files[i].hdr, name);
        if (e) {
            *file = &w->files[i];
            return e;
        }
    }
    return NULL;
}

static float half_to_float(uint16_t h) { /* IEEE 754 binary16 -> binary32 */
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, u;
    if (exp == 0) {
        if (man == 0) {
            u = sign; /* zero */
        } else { /* subnormal: shift the mantissa up until its leading bit appears */
            int e = 1;
            while (!(man & 0x400)) {
                man <<= 1;
                e--;
            }
            u = sign | ((uint32_t)(e + 112) << 23) | ((man & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (man << 13); /* inf / nan */
    } else {
        u = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* Load one tensor as float32 and check its shape. d1 = 0 means the tensor is 1-D. */
static float *load_tensor(const Weights *w, const char *name, int d0, int d1) {
    const STFile *f = NULL;
    const JVal *e = weights_find(w, name, &f);
    if (!e) die("tensor '%s' is not in the checkpoint\n", name);
    const JVal *dtype = j_get(e, "dtype"), *shape = j_get(e, "shape"), *offs = j_get(e, "data_offsets");
    if (!dtype || dtype->type != J_STR || !shape || shape->type != J_ARR || !offs || offs->type != J_ARR || offs->count != 2)
        die("tensor '%s': malformed header entry\n", name);

    size_t want_dims = d1 ? 2 : 1;
    int ok = shape->count == want_dims && (int)shape->items[0].num == d0 && (!d1 || (int)shape->items[1].num == d1);
    if (!ok) {
        fprintf(stderr, "error: tensor '%s' has shape [", name);
        for (size_t i = 0; i < shape->count; i++) fprintf(stderr, "%s%.0f", i ? ", " : "", shape->items[i].num);
        if (d1) fprintf(stderr, "], but config.json implies [%d, %d]\n", d0, d1);
        else fprintf(stderr, "], but config.json implies [%d]\n", d0);
        exit(1);
    }

    size_t count = (size_t)d0 * (size_t)(d1 ? d1 : 1);
    size_t beg = (size_t)offs->items[0].num, end = (size_t)offs->items[1].num;
    size_t elem = j_str_is(dtype, "F32") ? 4 : (j_str_is(dtype, "BF16") || j_str_is(dtype, "F16")) ? 2 : j_str_is(dtype, "F64") ? 8 : 0;
    if (!elem) die("tensor '%s': unsupported dtype %s\n", name, dtype->str);
    if (beg > end || end > f->data_len || end - beg != count * elem) die("tensor '%s': bad data offsets\n", name);

    const unsigned char *src = f->data + beg;
    float *out = alloc_floats(count);
    if (j_str_is(dtype, "F32")) {
        memcpy(out, src, count * 4);
    } else if (j_str_is(dtype, "BF16")) {
        /* bfloat16 keeps float32's sign and exponent and its top 7 mantissa
         * bits, so widening is a 16-bit shift: the dropped bits become zero. */
        for (size_t i = 0; i < count; i++) {
            uint16_t h;
            memcpy(&h, src + 2 * i, 2);
            uint32_t u = (uint32_t)h << 16;
            memcpy(&out[i], &u, 4);
        }
    } else if (j_str_is(dtype, "F16")) {
        for (size_t i = 0; i < count; i++) {
            uint16_t h;
            memcpy(&h, src + 2 * i, 2);
            out[i] = half_to_float(h);
        }
    } else {
        for (size_t i = 0; i < count; i++) {
            double d;
            memcpy(&d, src + 8 * i, 8);
            out[i] = (float)d;
        }
    }
    return out;
}

/* ========================================================================== *
 * 6. Thread pool and math kernels
 *
 * Almost all the time goes into "matrix times vector": each output element is
 * the dot product of one matrix row with the input vector. Rows are
 * independent, so they are spread over threads.
 *
 * The pool is deliberately small. One job at a time: a function plus a range
 * [0, n). Threads claim chunks of the range from a shared atomic counter until
 * it runs out, so a slow core just ends up claiming fewer chunks. Workers spin
 * for a moment between jobs before sleeping, because during generation the
 * next job arrives within microseconds.
 * ========================================================================== */

#if defined(__x86_64__) || defined(__i386__)
#define cpu_relax() __builtin_ia32_pause()
#define POOL_SPIN 3000
#elif defined(__aarch64__)
#define cpu_relax() __asm__ __volatile__("yield")
#define POOL_SPIN 30000
#else
#define cpu_relax() ((void)0)
#define POOL_SPIN 3000
#endif

typedef void (*JobFn)(void *ctx, int start, int end);

static struct {
    int n_threads; /* including the thread that calls pool_run */
    pthread_t *threads;
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    JobFn fn; /* the current job */
    void *ctx;
    int n, chunk;
    atomic_int next;    /* next unclaimed index of the current job */
    atomic_int gen;     /* job counter: a change means "new job available" */
    atomic_int pending; /* workers that have not finished the current job yet */
    atomic_int stop;
} pool;

static void pool_drain(void) { /* claim and run chunks until the job is exhausted */
    for (;;) {
        int start = atomic_fetch_add(&pool.next, pool.chunk);
        if (start >= pool.n) return;
        int end = start + pool.chunk;
        pool.fn(pool.ctx, start, end < pool.n ? end : pool.n);
    }
}

static void *pool_worker(void *arg) {
    (void)arg;
    int my_gen = 0;
    for (;;) {
        int spins = 0;
        while (atomic_load(&pool.gen) == my_gen && !atomic_load(&pool.stop)) {
            if (++spins < POOL_SPIN) {
                cpu_relax();
                continue;
            }
            pthread_mutex_lock(&pool.mu);
            while (atomic_load(&pool.gen) == my_gen && !atomic_load(&pool.stop)) pthread_cond_wait(&pool.cv_work, &pool.mu);
            pthread_mutex_unlock(&pool.mu);
        }
        if (atomic_load(&pool.stop)) return NULL;
        my_gen++;
        pool_drain();
        if (atomic_fetch_sub(&pool.pending, 1) == 1) { /* last worker out wakes the caller */
            pthread_mutex_lock(&pool.mu);
            pthread_cond_signal(&pool.cv_done);
            pthread_mutex_unlock(&pool.mu);
        }
    }
}

/* Run fn over [0, n) in pieces of `chunk`, on all threads. Returns when done. */
static void pool_run(JobFn fn, void *ctx, int n, int chunk) {
    if (pool.n_threads <= 1 || n <= chunk) {
        fn(ctx, 0, n);
        return;
    }
    pool.fn = fn;
    pool.ctx = ctx;
    pool.n = n;
    pool.chunk = chunk;
    atomic_store(&pool.next, 0);
    atomic_store(&pool.pending, pool.n_threads - 1);
    pthread_mutex_lock(&pool.mu);
    atomic_fetch_add(&pool.gen, 1); /* publishing the new generation releases the workers */
    pthread_cond_broadcast(&pool.cv_work);
    pthread_mutex_unlock(&pool.mu);

    pool_drain(); /* the calling thread works too */

    int spins = 0;
    while (atomic_load(&pool.pending) != 0) {
        if (++spins < POOL_SPIN) {
            cpu_relax();
            continue;
        }
        pthread_mutex_lock(&pool.mu);
        while (atomic_load(&pool.pending) != 0) pthread_cond_wait(&pool.cv_done, &pool.mu);
        pthread_mutex_unlock(&pool.mu);
    }
}

static int default_thread_count(void) {
#if defined(__APPLE__)
    /* Apple Silicon has fast and slow cores; use only the fast ones. */
    int n = 0;
    size_t len = sizeof n;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, NULL, 0) == 0 && n > 0) return n;
#endif
    long n_online = sysconf(_SC_NPROCESSORS_ONLN);
    return n_online > 0 ? (int)n_online : 1;
}

static void pool_init(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 256) n_threads = 256;
    pool.n_threads = n_threads;
    pthread_mutex_init(&pool.mu, NULL);
    pthread_cond_init(&pool.cv_work, NULL);
    pthread_cond_init(&pool.cv_done, NULL);
    pool.threads = xmalloc((size_t)n_threads * sizeof(pthread_t));
    for (int i = 0; i < n_threads - 1; i++)
        if (pthread_create(&pool.threads[i], NULL, pool_worker, NULL) != 0) die("cannot create thread\n");
}

static void pool_shutdown(void) {
    pthread_mutex_lock(&pool.mu);
    atomic_store(&pool.stop, 1);
    pthread_cond_broadcast(&pool.cv_work);
    pthread_mutex_unlock(&pool.mu);
    for (int i = 0; i < pool.n_threads - 1; i++) pthread_join(pool.threads[i], NULL);
    free(pool.threads);
}

/* --- dot product ----------------------------------------------------------
 *
 * Written with GCC/Clang vector types so the compiler emits SIMD (NEON on
 * Apple Silicon, AVX on x86) without any per-CPU code here. Four independent
 * accumulators let the CPU overlap consecutive multiply-adds instead of
 * waiting for each one to finish.
 */
#if defined(__GNUC__) || defined(__clang__)
typedef float v8f __attribute__((vector_size(32)));

static inline v8f v8_load(const float *p) {
    v8f v;
    memcpy(&v, p, sizeof v); /* an unaligned load; compiles to one instruction */
    return v;
}

static float dot(const float *a, const float *b, int n) {
    v8f s0 = {0}, s1 = {0}, s2 = {0}, s3 = {0};
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 += v8_load(a + i) * v8_load(b + i);
        s1 += v8_load(a + i + 8) * v8_load(b + i + 8);
        s2 += v8_load(a + i + 16) * v8_load(b + i + 16);
        s3 += v8_load(a + i + 24) * v8_load(b + i + 24);
    }
    for (; i + 8 <= n; i += 8) s0 += v8_load(a + i) * v8_load(b + i);
    s0 = (s0 + s1) + (s2 + s3);
    float s = ((s0[0] + s0[1]) + (s0[2] + s0[3])) + ((s0[4] + s0[5]) + (s0[6] + s0[7]));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#else
static float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif

/* y = W x, where W is [rows x cols] stored row by row (the layout PyTorch's nn.Linear uses). */
typedef struct {
    float *y;
    const float *W, *x;
    int cols;
} MatvecJob;

static void matvec_job(void *ctx, int r0, int r1) {
    const MatvecJob *j = ctx;
    for (int r = r0; r < r1; r++) j->y[r] = dot(j->W + (size_t)r * (size_t)j->cols, j->x, j->cols);
}

static void matvec(float *y, const float *W, const float *x, int rows, int cols) {
    MatvecJob j = {y, W, x, cols};
    int chunk = rows / (pool.n_threads * 4); /* a few chunks per thread evens out uneven cores */
    int min_chunk = 32768 / cols + 1;        /* but keep each chunk worth the hand-off */
    if (chunk < min_chunk) chunk = min_chunk;
    pool_run(matvec_job, &j, rows, chunk);
}

/* ========================================================================== *
 * 7. The model
 *
 * Qwen3 is a decoder-only transformer. For one token at position `pos`:
 *
 *     x = embedding[token]
 *     for each layer:
 *         x = x + attention(rmsnorm(x))      "look at earlier tokens"
 *         x = x + mlp(rmsnorm(x))            "think about what you saw"
 *     logits = lm_head(rmsnorm(x))           one score per vocabulary entry
 *
 * The engine processes one token at a time and keeps each layer's attention
 * keys and values for all earlier positions (the "KV cache"), so producing
 * token N costs one pass through the layers, not N.
 * ========================================================================== */

typedef struct {
    int vocab_size;
    int dim;        /* width of the residual stream (hidden_size) */
    int hidden;     /* width inside the MLP (intermediate_size) */
    int n_layers;
    int n_heads;    /* query heads */
    int n_kv_heads; /* key/value heads; fewer than n_heads = grouped-query attention */
    int head_dim;
    int max_pos;    /* longest sequence the model was configured for */
    int tie_embeddings;
    float rms_eps, rope_theta;
} Config;

typedef struct {
    float *attn_norm;          /* [dim]                          RMSNorm before attention */
    float *wq;                 /* [n_heads*head_dim, dim]         query projection */
    float *wk, *wv;            /* [n_kv_heads*head_dim, dim]      key and value projections */
    float *wo;                 /* [dim, n_heads*head_dim]         output projection */
    float *q_norm, *k_norm;    /* [head_dim]                      RMSNorm on each query / key head */
    float *mlp_norm;           /* [dim]                          RMSNorm before the MLP */
    float *w_gate, *w_up;      /* [hidden, dim] */
    float *w_down;             /* [dim, hidden] */
} Layer;

typedef struct {
    Config cfg;
    int ctx; /* how many positions the KV cache can hold */

    float *tok_emb;    /* [vocab_size, dim] */
    Layer *layers;
    float *final_norm; /* [dim] */
    float *lm_head;    /* [vocab_size, dim]; the same array as tok_emb when tied */

    float *rope_cos, *rope_sin; /* [ctx, head_dim/2] */
    float **k_cache, **v_cache; /* per layer: [ctx, n_kv_heads*head_dim] */

    /* scratch buffers for one forward pass */
    float *x, *xn, *q, *attn_out, *tmp, *hg, *hu, *att, *logits;
} Model;

static void config_load(Config *c, const char *dir) {
    char *path = path_join(dir, "config.json");
    size_t len;
    char *text = read_file(path, &len);
    Arena arena = {0};
    JVal root;
    json_parse(text, len, &arena, &root);

    memset(c, 0, sizeof *c);
    c->vocab_size = (int)j_num(&root, "vocab_size", 0);
    c->dim = (int)j_num(&root, "hidden_size", 0);
    c->hidden = (int)j_num(&root, "intermediate_size", 0);
    c->n_layers = (int)j_num(&root, "num_hidden_layers", 0);
    c->n_heads = (int)j_num(&root, "num_attention_heads", 0);
    c->n_kv_heads = (int)j_num(&root, "num_key_value_heads", c->n_heads);
    c->head_dim = (int)j_num(&root, "head_dim", c->n_heads ? c->dim / c->n_heads : 0);
    c->max_pos = (int)j_num(&root, "max_position_embeddings", 32768);
    c->rms_eps = (float)j_num(&root, "rms_norm_eps", 1e-6);
    const JVal *tie = j_get(&root, "tie_word_embeddings");
    c->tie_embeddings = tie && tie->type == J_TRUE;

    /* rope_theta sits at the top level in older configs and inside rope_parameters in newer ones */
    const JVal *theta = j_get(&root, "rope_theta");
    if (!theta || theta->type != J_NUM) theta = j_get(j_get(&root, "rope_parameters"), "rope_theta");
    if (theta && theta->type == J_NUM) {
        c->rope_theta = (float)theta->num;
    } else {
        fprintf(stderr, "warning: %s has no rope_theta; assuming 10000\n", path);
        c->rope_theta = 10000.0f;
    }

    const JVal *act = j_get(&root, "hidden_act");
    if (act && !j_str_is(act, "silu")) die("%s: hidden_act '%s' is not supported (only silu)\n", path, act->str);
    const JVal *bias = j_get(&root, "attention_bias");
    if (bias && bias->type == J_TRUE) die("%s: attention_bias=true is not supported\n", path);
    const JVal *type = j_get(&root, "model_type");
    if (type && !j_str_is(type, "qwen3"))
        fprintf(stderr, "warning: model_type is '%s'; this engine implements 'qwen3'\n", type->str);

    if (c->vocab_size <= 0 || c->dim <= 0 || c->hidden <= 0 || c->n_layers <= 0 || c->n_heads <= 0 || c->n_kv_heads <= 0 ||
        c->head_dim <= 0)
        die("%s: missing or invalid model dimensions\n", path);
    if (c->n_heads % c->n_kv_heads != 0) die("%s: num_attention_heads must be a multiple of num_key_value_heads\n", path);
    if (c->head_dim % 2 != 0) die("%s: head_dim must be even\n", path);

    arena_free(&arena);
    free(text);
    free(path);
}

static void model_load(Model *m, const char *dir, int ctx) {
    Config *c = &m->cfg;
    int D = c->dim, QD = c->n_heads * c->head_dim, KD = c->n_kv_heads * c->head_dim;
    if (ctx > c->max_pos) ctx = c->max_pos;
    m->ctx = ctx;

    Weights w;
    weights_open(&w, dir);
    char name[256];

    m->tok_emb = load_tensor(&w, "model.embed_tokens.weight", c->vocab_size, D);
    m->layers = xcalloc((size_t)c->n_layers, sizeof(Layer));
    for (int l = 0; l < c->n_layers; l++) {
        Layer *L = &m->layers[l];
#define LOAD(field, suffix, d0, d1)                                       \
    snprintf(name, sizeof name, "model.layers.%d." suffix ".weight", l); \
    L->field = load_tensor(&w, name, d0, d1)
        LOAD(attn_norm, "input_layernorm", D, 0);
        LOAD(wq, "self_attn.q_proj", QD, D);
        LOAD(wk, "self_attn.k_proj", KD, D);
        LOAD(wv, "self_attn.v_proj", KD, D);
        LOAD(wo, "self_attn.o_proj", D, QD);
        LOAD(q_norm, "self_attn.q_norm", c->head_dim, 0);
        LOAD(k_norm, "self_attn.k_norm", c->head_dim, 0);
        LOAD(mlp_norm, "post_attention_layernorm", D, 0);
        LOAD(w_gate, "mlp.gate_proj", c->hidden, D);
        LOAD(w_up, "mlp.up_proj", c->hidden, D);
        LOAD(w_down, "mlp.down_proj", D, c->hidden);
#undef LOAD
    }
    m->final_norm = load_tensor(&w, "model.norm.weight", D, 0);

    /* The output layer maps the final vector back to one score per token. In
     * Qwen3-0.6B it shares its matrix with the input embedding ("tied"). */
    const STFile *unused;
    if (c->tie_embeddings || !weights_find(&w, "lm_head.weight", &unused)) m->lm_head = m->tok_emb;
    else m->lm_head = load_tensor(&w, "lm_head.weight", c->vocab_size, D);
    weights_close(&w);

    /* Rotary position embedding tables: one rotation angle per (position, frequency).
     * Frequencies fall off geometrically, from 1 radian per token down to 1/theta. */
    int half = c->head_dim / 2;
    m->rope_cos = alloc_floats((size_t)ctx * half);
    m->rope_sin = alloc_floats((size_t)ctx * half);
    for (int i = 0; i < half; i++) {
        float inv_freq = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)c->head_dim);
        for (int p = 0; p < ctx; p++) {
            float angle = (float)p * inv_freq;
            m->rope_cos[(size_t)p * half + i] = cosf(angle);
            m->rope_sin[(size_t)p * half + i] = sinf(angle);
        }
    }

    /* KV cache. calloc hands back untouched zero pages, so only the positions
     * actually used ever take real memory. */
    m->k_cache = xmalloc((size_t)c->n_layers * sizeof(float *));
    m->v_cache = xmalloc((size_t)c->n_layers * sizeof(float *));
    for (int l = 0; l < c->n_layers; l++) {
        m->k_cache[l] = xcalloc((size_t)ctx * KD, sizeof(float));
        m->v_cache[l] = xcalloc((size_t)ctx * KD, sizeof(float));
    }

    m->x = alloc_floats(D);
    m->xn = alloc_floats(D);
    m->tmp = alloc_floats(D);
    m->q = alloc_floats(QD);
    m->attn_out = alloc_floats(QD);
    m->hg = alloc_floats(c->hidden);
    m->hu = alloc_floats(c->hidden);
    m->att = alloc_floats((size_t)c->n_heads * ctx);
    m->logits = alloc_floats(c->vocab_size);
}

/* RMSNorm: scale the vector to unit root-mean-square, then apply a learned
 * per-element gain. Works in place (out == x). */
static void rmsnorm(float *out, const float *x, const float *gain, int n, float eps) {
    float inv = 1.0f / sqrtf(dot(x, x, n) / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * gain[i];
}

/* RoPE: encode position by rotating pairs of coordinates. Element i is paired
 * with element i + half, and the pair is rotated by pos * freq[i]. Rotating
 * queries and keys this way makes their dot product depend only on how far
 * apart the two tokens are, which is how the model knows word order. */
static void rope(float *v, const float *cs, const float *sn, int half) {
    for (int i = 0; i < half; i++) {
        float a = v[i], b = v[i + half];
        v[i] = a * cs[i] - b * sn[i];
        v[i + half] = b * cs[i] + a * sn[i];
    }
}

typedef struct {
    Model *m;
    int layer, pos;
} AttnJob;

/* Attention for query heads [h0, h1): score every cached position, softmax,
 * and take the weighted average of the cached values. */
static void attn_job(void *ctx, int h0, int h1) {
    const AttnJob *j = ctx;
    Model *m = j->m;
    const Config *c = &m->cfg;
    int hd = c->head_dim, KD = c->n_kv_heads * hd, group = c->n_heads / c->n_kv_heads, T = j->pos + 1;
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = h0; h < h1; h++) {
        /* grouped-query attention: several query heads share one key/value head */
        const float *q = m->q + (size_t)h * hd;
        const float *K = m->k_cache[j->layer] + (size_t)(h / group) * hd;
        const float *V = m->v_cache[j->layer] + (size_t)(h / group) * hd;
        float *att = m->att + (size_t)h * m->ctx;

        float max = -INFINITY;
        for (int t = 0; t < T; t++) {
            att[t] = dot(q, K + (size_t)t * KD, hd) * scale;
            if (att[t] > max) max = att[t];
        }
        float sum = 0.0f;
        for (int t = 0; t < T; t++) {
            att[t] = expf(att[t] - max);
            sum += att[t];
        }
        float inv = 1.0f / sum;

        float *out = m->attn_out + (size_t)h * hd;
        memset(out, 0, (size_t)hd * sizeof(float));
        for (int t = 0; t < T; t++) {
            float wgt = att[t] * inv;
            const float *v = V + (size_t)t * KD;
            for (int i = 0; i < hd; i++) out[i] += wgt * v[i];
        }
    }
}

/* Run one token through the model at position `pos`, extending the KV cache.
 * Returns the logits for the next token, or NULL if want_logits is 0 (prompt
 * tokens other than the last do not need them, and the output layer is by far
 * the largest single matrix). */
static float *forward(Model *m, int token, int pos, int want_logits) {
    const Config *c = &m->cfg;
    int D = c->dim, hd = c->head_dim, half = hd / 2;
    int QD = c->n_heads * hd, KD = c->n_kv_heads * hd;
    const float *cs = m->rope_cos + (size_t)pos * half, *sn = m->rope_sin + (size_t)pos * half;
    float *x = m->x;

    memcpy(x, m->tok_emb + (size_t)token * D, (size_t)D * sizeof(float));

    for (int l = 0; l < c->n_layers; l++) {
        const Layer *L = &m->layers[l];

        /* ---- attention ---- */
        rmsnorm(m->xn, x, L->attn_norm, D, c->rms_eps);
        float *k = m->k_cache[l] + (size_t)pos * KD; /* this position's keys and values are */
        float *v = m->v_cache[l] + (size_t)pos * KD; /* written straight into the cache */
        matvec(m->q, L->wq, m->xn, QD, D);
        matvec(k, L->wk, m->xn, KD, D);
        matvec(v, L->wv, m->xn, KD, D);
        for (int h = 0; h < c->n_heads; h++) { /* Qwen3 normalizes each query and key head, then rotates */
            rmsnorm(m->q + h * hd, m->q + h * hd, L->q_norm, hd, c->rms_eps);
            rope(m->q + h * hd, cs, sn, half);
        }
        for (int g = 0; g < c->n_kv_heads; g++) {
            rmsnorm(k + g * hd, k + g * hd, L->k_norm, hd, c->rms_eps);
            rope(k + g * hd, cs, sn, half);
        }
        AttnJob job = {m, l, pos};
        if (pos >= 256) pool_run(attn_job, &job, c->n_heads, 1); /* worth threading once the history is long */
        else attn_job(&job, 0, c->n_heads);
        matvec(m->tmp, L->wo, m->attn_out, D, QD);
        for (int i = 0; i < D; i++) x[i] += m->tmp[i];

        /* ---- MLP (SwiGLU): down( silu(gate(x)) * up(x) ) ---- */
        rmsnorm(m->xn, x, L->mlp_norm, D, c->rms_eps);
        matvec(m->hg, L->w_gate, m->xn, c->hidden, D);
        matvec(m->hu, L->w_up, m->xn, c->hidden, D);
        for (int i = 0; i < c->hidden; i++) m->hg[i] = m->hg[i] / (1.0f + expf(-m->hg[i])) * m->hu[i];
        matvec(m->tmp, L->w_down, m->hg, D, c->hidden);
        for (int i = 0; i < D; i++) x[i] += m->tmp[i];
    }

    if (!want_logits) return NULL;
    rmsnorm(m->xn, x, m->final_norm, D, c->rms_eps);
    matvec(m->logits, m->lm_head, m->xn, c->vocab_size, D);
    return m->logits;
}

/* ========================================================================== *
 * 8. Sampler
 *
 * The logits are unnormalized scores. To pick the next token:
 *   temperature  divide the scores; below 1 sharpens the distribution
 *   top-k        keep only the k highest-scoring tokens
 *   top-p        of those, keep the smallest set whose probability reaches p
 * then draw one at random from what is left.
 * ========================================================================== */

typedef struct {
    float temperature, top_p, presence_penalty;
    int top_k;
    uint64_t rng;
} Sampler;

typedef struct {
    float p;
    int id;
} Cand;

static uint64_t rng_next(uint64_t *s) { /* xorshift64* */
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static float rng_uniform(uint64_t *s) { return (float)(rng_next(s) >> 40) * (1.0f / 16777216.0f); } /* [0, 1) */

static int cand_cmp_desc(const void *a, const void *b) {
    float x = ((const Cand *)a)->p, y = ((const Cand *)b)->p;
    return (x < y) - (x > y);
}

static int sample(Sampler *s, const float *logits, int n, Cand *cand) {
    if (s->temperature <= 0.0f) { /* greedy: always the most likely token */
        int best = 0;
        for (int i = 1; i < n; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    /* top-k: collect the k largest logits, sorted from high to low */
    int k = (s->top_k > 0 && s->top_k < n) ? s->top_k : n, m = 0;
    if (k < n) {
        for (int i = 0; i < n; i++) {
            float v = logits[i];
            if (m == k && v <= cand[k - 1].p) continue; /* the common case: not in the top k */
            int j = (m < k) ? m++ : k - 1;
            while (j > 0 && cand[j - 1].p < v) {
                cand[j] = cand[j - 1];
                j--;
            }
            cand[j].p = v;
            cand[j].id = i;
        }
    } else {
        for (int i = 0; i < n; i++) {
            cand[i].p = logits[i];
            cand[i].id = i;
        }
        qsort(cand, (size_t)n, sizeof(Cand), cand_cmp_desc);
        m = n;
    }

    /* temperature + softmax over the survivors */
    float max = cand[0].p, sum = 0.0f;
    for (int i = 0; i < m; i++) {
        cand[i].p = expf((cand[i].p - max) / s->temperature);
        sum += cand[i].p;
    }

    /* top-p: cut the tail once the running total reaches p */
    if (s->top_p > 0.0f && s->top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < m; i++) {
            cum += cand[i].p;
            if (cum >= s->top_p * sum) {
                m = i + 1;
                sum = cum;
                break;
            }
        }
    }

    float r = rng_uniform(&s->rng) * sum;
    for (int i = 0; i < m; i++) {
        r -= cand[i].p;
        if (r < 0.0f) return cand[i].id;
    }
    return cand[m - 1].id;
}

/* ========================================================================== *
 * 9. Chat loop and command line
 *
 * Qwen3 was trained on conversations written in the ChatML format:
 *
 *     <|im_start|>system
 *     You are a helpful assistant.<|im_end|>
 *     <|im_start|>user
 *     Why is the sky blue?<|im_end|>
 *     <|im_start|>assistant
 *     <think>
 *     ...the model's reasoning...
 *     </think>
 *
 *     Because of Rayleigh scattering...<|im_end|>
 *
 * The engine writes everything up to "<|im_start|>assistant\n" and the model
 * continues from there until it emits <|im_end|>. Thinking can be switched off
 * by pre-filling an empty "<think>\n\n</think>\n\n" block for it.
 * ========================================================================== */

typedef struct {
    const char *model_dir, *prompt, *system;
    int max_new, ctx, threads, think, raw, stats;
    float temperature, top_p, presence_penalty;
    int top_k;
    uint64_t seed;
    int have_seed;
    /* tools for checking the engine against a reference implementation */
    const char *encode_text, *encode_json, *dump_ids, *dump_path, *log_path, *replay_path;
    int parse_special;
} Options;

static volatile sig_atomic_t g_interrupted = 0;
static void on_sigint(int sig) {
    (void)sig;
    g_interrupted = 1;
}

typedef struct {
    Model *m;
    Tokenizer *t;
    Sampler s;
    const Options *o;
    Cand *cand;          /* sampler scratch */
    uint8_t *seen;       /* tokens used in the current reply (presence penalty) */
    int pos;             /* number of positions in the KV cache */
    int use_color;
    FILE *log;           /* --log-tokens: every (position, token) written to the cache */
    IntVec *replay;      /* --replay: scripted replies used instead of sampling */
    int n_replay, replay_turn, replay_i;
    /* timing */
    int n_prompt, n_gen;
    double t_prompt, t_gen;
} Chat;

/* Put one token into the next cache position. */
static float *step(Chat *ch, int id, int want_logits) {
    if (ch->log) fprintf(ch->log, "%d %d\n", ch->pos, id);
    return forward(ch->m, id, ch->pos++, want_logits);
}

/* Feed tokens through the model. Only the last one produces logits. */
static float *feed(Chat *ch, const int *toks, int n) {
    float *logits = NULL;
    double t0 = now_sec();
    for (int i = 0; i < n; i++) {
        if (ch->pos >= ch->m->ctx) return NULL;
        logits = step(ch, toks[i], i == n - 1);
    }
    ch->n_prompt += n;
    ch->t_prompt += now_sec() - t0;
    return logits;
}

/* Generate until the model ends its turn. The tokens it produced go to `reply`. */
static void generate(Chat *ch, float *logits, IntVec *reply) {
    const Tokenizer *t = ch->t;
    int vocab = ch->m->cfg.vocab_size;
    double t0 = now_sec();
    int produced = 0;
    for (int n = 0; n < ch->o->max_new; n++) {
        if (g_interrupted) {
            fprintf(stderr, "\n[interrupted]");
            break;
        }
        if (ch->s.presence_penalty != 0.0f) {
            /* Discourage tokens this reply has already used. `seen` makes sure a
             * token that occurred several times is still only penalized once. */
            for (int i = 0; i < reply->n; i++) {
                int id = reply->v[i];
                if (!ch->seen[id]) {
                    logits[id] -= ch->s.presence_penalty;
                    ch->seen[id] = 1;
                }
            }
            for (int i = 0; i < reply->n; i++) ch->seen[reply->v[i]] = 0;
        }
        int id;
        if (ch->replay) { /* testing: take the next scripted token instead of sampling */
            const IntVec *script = ch->replay_turn < ch->n_replay ? &ch->replay[ch->replay_turn] : NULL;
            id = (script && ch->replay_i < script->n) ? script->v[ch->replay_i++] : t->im_end;
        } else {
            id = sample(&ch->s, logits, vocab, ch->cand);
        }
        if (id == t->im_end || id == t->endoftext) break; /* the model ended its turn */

        iv_push(reply, id);
        size_t len;
        const unsigned char *bytes = tok_bytes(t, id, &len);
        if (ch->use_color && id == t->think) fputs("\x1b[2m", stdout); /* dim the reasoning */
        fwrite(bytes, 1, len, stdout);
        if (ch->use_color && id == t->think_end) fputs("\x1b[0m", stdout);
        fflush(stdout);

        if (ch->pos >= ch->m->ctx) {
            fprintf(stderr, "\n[context window full; raise it with -c]");
            break;
        }
        logits = step(ch, id, 1);
        produced++;
    }
    ch->replay_turn++;
    ch->replay_i = 0;
    if (ch->use_color) fputs("\x1b[0m", stdout);
    fputc('\n', stdout);
    fflush(stdout);
    ch->n_gen += produced;
    ch->t_gen += now_sec() - t0;
}

static void print_stats(Chat *ch) {
    fprintf(stderr, "[prompt: %d tokens, %.1f tok/s | generated: %d tokens, %.1f tok/s | context used: %d/%d]\n", ch->n_prompt,
            ch->t_prompt > 0 ? ch->n_prompt / ch->t_prompt : 0.0, ch->n_gen, ch->t_gen > 0 ? ch->n_gen / ch->t_gen : 0.0,
            ch->pos, ch->m->ctx);
    ch->n_prompt = ch->n_gen = 0;
    ch->t_prompt = ch->t_gen = 0;
}

/* Append "<|im_start|>{role}\n{text}<|im_end|>\n" to a token list. */
static void push_turn(const Tokenizer *t, IntVec *out, const char *role, const char *text) {
    size_t rl = strlen(role), tl = strlen(text);
    char *body = xmalloc(rl + tl + 2);
    memcpy(body, role, rl);
    body[rl] = '\n';
    memcpy(body + rl + 1, text, tl + 1);
    iv_push(out, t->im_start);
    tok_encode(t, body, rl + 1 + tl, 0, out); /* user text is never parsed for special tokens */
    iv_push(out, t->im_end);
    tok_encode_cstr(t, "\n", out);
    free(body);
}

static void run_chat(Chat *ch) {
    const Options *o = ch->o;
    const Tokenizer *t = ch->t;
    int interactive = (o->prompt == NULL);
    IntVec in = {0}, reply = {0}, header = {0};
    int have_reply = 0; /* is there a previous assistant reply that still has to be written into history? */
    int body_pos = 0;   /* cache position right after that reply's "<|im_start|>assistant\n" */
    int first = 1;
    char *line = NULL;
    size_t line_cap = 0;

    iv_push(&header, t->im_start);
    tok_encode_cstr(t, "assistant\n", &header);

    if (interactive) fprintf(stderr, "Type a message. /reset clears the conversation, /exit quits.\n");

    for (;;) {
        const char *user;
        if (interactive) {
            fputs("\n> ", stderr);
            fflush(stderr);
            g_interrupted = 0;
            ssize_t n = getline(&line, &line_cap, stdin);
            if (n < 0) break; /* end of input, or Ctrl-C at the prompt */
            while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
            if (n == 0) continue;
            if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) break;
            if (!strcmp(line, "/reset")) {
                ch->pos = 0; /* forgetting everything is just rewinding the cache */
                have_reply = 0;
                first = 1;
                fprintf(stderr, "[conversation cleared]\n");
                continue;
            }
            user = line;
        } else {
            user = o->prompt;
        }

        in.n = 0;
        if (first && o->system && o->system[0]) push_turn(t, &in, "system", o->system);
        first = 0;

        /* Write the previous reply into history the way the chat template does:
         * the answer only, without its <think> block. The cache currently holds
         * the reply as it was generated, thinking included, so rewind to just
         * after "<|im_start|>assistant\n" and feed the cleaned-up answer again.
         * Rewinding is free: cache entries past `pos` are simply overwritten. */
        if (have_reply) {
            int start = 0;
            for (int i = 0; i < reply.n; i++)
                if (reply.v[i] == t->think_end) start = i + 1;
            if (start > 0) { /* the template also drops newlines right after </think> */
                while (start < reply.n) {
                    size_t len;
                    const unsigned char *b = tok_bytes(t, reply.v[start], &len);
                    size_t k = 0;
                    while (k < len && b[k] == '\n') k++;
                    if (k < len) break;
                    start++;
                }
            }
            if (start > 0 || !o->think) {
                ch->pos = body_pos;
                iv_append(&in, reply.v + start, reply.n - start);
            } /* else: the cache already holds exactly the tokens history needs */
            iv_push(&in, t->im_end);
            tok_encode_cstr(t, "\n", &in);
        }

        push_turn(t, &in, "user", user);
        iv_append(&in, header.v, header.n);
        body_pos = ch->pos + in.n;
        if (!o->think) { /* an empty reasoning block tells the model to answer directly */
            iv_push(&in, t->think);
            tok_encode_cstr(t, "\n\n", &in);
            iv_push(&in, t->think_end);
            tok_encode_cstr(t, "\n\n", &in);
        }

        float *logits = feed(ch, in.v, in.n);
        if (!logits) {
            fprintf(stderr, "[the conversation no longer fits in the %d-token context; use /reset or a larger -c]\n", ch->m->ctx);
            if (!interactive) break;
            ch->pos = 0;
            have_reply = 0;
            first = 1;
            continue;
        }
        reply.n = 0;
        g_interrupted = 0;
        generate(ch, logits, &reply);
        have_reply = 1;
        if (o->stats || !interactive) print_stats(ch);
        if (!interactive) break;
    }
    free(line);
    free(in.v);
    free(reply.v);
    free(header.v);
}

/* --raw: no chat template. The prompt is fed as written (special tokens in it
 * are honoured) and the model simply continues the text. */
static void run_raw(Chat *ch) {
    IntVec in = {0}, reply = {0};
    tok_encode(ch->t, ch->o->prompt, strlen(ch->o->prompt), 1, &in);
    if (in.n == 0) die("--raw needs a non-empty prompt (-p)\n");
    float *logits = feed(ch, in.v, in.n);
    if (!logits) die("the prompt does not fit in the %d-token context\n", ch->m->ctx);
    fputs(ch->o->prompt, stdout);
    generate(ch, logits, &reply);
    print_stats(ch);
    free(in.v);
    free(reply.v);
}

static void print_ids(const IntVec *v) {
    for (int i = 0; i < v->n; i++) printf(i ? " %d" : "%d", v->v[i]);
    printf("\n");
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [options]\n"
            "\n"
            "  -m DIR            model directory holding config.json, tokenizer.json and\n"
            "                    model.safetensors                            (default: model)\n"
            "  -p TEXT           answer this one prompt and exit; without -p, start a chat\n"
            "  -s TEXT           system prompt\n"
            "  -n N              maximum new tokens per reply                 (default: 2048)\n"
            "  -c N              context window in tokens                     (default: 4096)\n"
            "  --no-think        answer directly, without the reasoning block\n"
            "  --think           let the model reason first                   (default)\n"
            "  -t F              temperature; 0 = always pick the top token   (default: 0.6, or 0.7 with --no-think)\n"
            "  -k N              top-k; 0 = off                               (default: 20)\n"
            "  -P F              top-p                                        (default: 0.95, or 0.8 with --no-think)\n"
            "  --presence-penalty F   penalize tokens already used in the reply (default: 0)\n"
            "  --seed N          random seed                                  (default: from the clock)\n"
            "  --threads N       worker threads                               (default: performance cores)\n"
            "  --raw             no chat template: continue the text given with -p\n"
            "  --stats           print speed after every reply\n"
            "\n"
            "verification tools:\n"
            "  --encode TEXT           print the token ids of TEXT and exit\n"
            "  --encode-json FILE      FILE is a JSON array of strings; print one line of ids per string\n"
            "  --parse-special         with the two options above, recognise <|im_start|> etc. in the text\n"
            "  --dump-logits IDS FILE  feed the comma-separated token IDS and write the logits after\n"
            "                          every one of them to FILE as raw float32\n"
            "  --log-tokens FILE       record every \"position token\" pair written to the context\n"
            "  --replay FILE           instead of sampling, emit the token ids on line N of FILE as reply N\n",
            argv0);
}

static const char *need_value(int argc, char **argv, int *i) {
    if (*i + 1 >= argc) die("option %s needs a value\n", argv[*i]);
    return argv[++*i];
}

int main(int argc, char **argv) {
    Options o = {0};
    o.model_dir = "model";
    o.max_new = 2048;
    o.ctx = 4096;
    o.think = 1;
    o.temperature = -1.0f; /* negative = not set on the command line */
    o.top_p = -1.0f;
    o.top_k = 20;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-m")) o.model_dir = need_value(argc, argv, &i);
        else if (!strcmp(a, "-p")) o.prompt = need_value(argc, argv, &i);
        else if (!strcmp(a, "-s")) o.system = need_value(argc, argv, &i);
        else if (!strcmp(a, "-n")) o.max_new = atoi(need_value(argc, argv, &i));
        else if (!strcmp(a, "-c")) o.ctx = atoi(need_value(argc, argv, &i));
        else if (!strcmp(a, "-t")) o.temperature = (float)atof(need_value(argc, argv, &i));
        else if (!strcmp(a, "-k")) o.top_k = atoi(need_value(argc, argv, &i));
        else if (!strcmp(a, "-P")) o.top_p = (float)atof(need_value(argc, argv, &i));
        else if (!strcmp(a, "--presence-penalty")) o.presence_penalty = (float)atof(need_value(argc, argv, &i));
        else if (!strcmp(a, "--seed")) {
            o.seed = strtoull(need_value(argc, argv, &i), NULL, 10);
            o.have_seed = 1;
        } else if (!strcmp(a, "--threads")) o.threads = atoi(need_value(argc, argv, &i));
        else if (!strcmp(a, "--think")) o.think = 1;
        else if (!strcmp(a, "--no-think")) o.think = 0;
        else if (!strcmp(a, "--raw")) o.raw = 1;
        else if (!strcmp(a, "--stats")) o.stats = 1;
        else if (!strcmp(a, "--encode")) o.encode_text = need_value(argc, argv, &i);
        else if (!strcmp(a, "--encode-json")) o.encode_json = need_value(argc, argv, &i);
        else if (!strcmp(a, "--parse-special")) o.parse_special = 1;
        else if (!strcmp(a, "--dump-logits")) {
            o.dump_ids = need_value(argc, argv, &i);
            o.dump_path = need_value(argc, argv, &i);
        } else if (!strcmp(a, "--log-tokens")) o.log_path = need_value(argc, argv, &i);
        else if (!strcmp(a, "--replay")) o.replay_path = need_value(argc, argv, &i);
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            die("unknown option %s\n", a);
        }
    }
    if (o.max_new < 1) die("-n must be at least 1\n");
    if (o.ctx < 16) die("-c must be at least 16\n");
    if (o.raw && !o.prompt) die("--raw needs a prompt (-p)\n");

    { /* everything here assumes a little-endian machine, as the file formats do */
        uint16_t probe = 1;
        if (*(unsigned char *)&probe != 1) die("big-endian machines are not supported\n");
    }

    char *tok_path = path_join(o.model_dir, "tokenizer.json");
    Tokenizer tok;

    /* Tokenizer-only modes: no weights needed. */
    if (o.encode_text || o.encode_json) {
        tok_load(&tok, tok_path, 0);
        IntVec ids = {0};
        if (o.encode_text) {
            tok_encode(&tok, o.encode_text, strlen(o.encode_text), o.parse_special, &ids);
            print_ids(&ids);
        } else {
            size_t len;
            char *text = read_file(o.encode_json, &len);
            Arena arena = {0};
            JVal root;
            json_parse(text, len, &arena, &root);
            if (root.type != J_ARR) die("%s: expected a JSON array of strings\n", o.encode_json);
            for (size_t i = 0; i < root.count; i++) {
                if (root.items[i].type != J_STR) die("%s: element %zu is not a string\n", o.encode_json, i);
                ids.n = 0;
                tok_encode(&tok, root.items[i].str, root.items[i].len, o.parse_special, &ids);
                print_ids(&ids);
            }
        }
        return 0;
    }

    Model model;
    memset(&model, 0, sizeof model);
    double t0 = now_sec();
    config_load(&model.cfg, o.model_dir);
    if (!o.dump_ids) { /* --dump-logits works on token ids and needs no tokenizer */
        tok_load(&tok, tok_path, model.cfg.vocab_size);
        if (tok.im_start < 0 || tok.im_end < 0 || tok.endoftext < 0)
            die("%s: the ChatML tokens <|im_start|>/<|im_end|> are missing\n", tok_path);
        if (tok.think < 0 || tok.think_end < 0) die("%s: the <think> tokens are missing; is this a Qwen3 tokenizer?\n", tok_path);
    }
    pool_init(o.threads > 0 ? o.threads : default_thread_count());
    model_load(&model, o.model_dir, o.ctx);
    const Config *c = &model.cfg;
    fprintf(stderr, "loaded %s in %.1fs: %d layers, dim %d, %d heads (%d kv), vocab %d, context %d, %d threads\n", o.model_dir,
            now_sec() - t0, c->n_layers, c->dim, c->n_heads, c->n_kv_heads, c->vocab_size, model.ctx, pool.n_threads);

    /* Verification mode: raw logits for a fixed token sequence. */
    if (o.dump_ids) {
        FILE *f = fopen(o.dump_path, "wb");
        if (!f) die("cannot write %s\n", o.dump_path);
        int pos = 0;
        for (const char *p = o.dump_ids; *p;) {
            char *end;
            long id = strtol(p, &end, 10);
            if (end == p) die("--dump-logits: expected comma-separated token ids\n");
            if (id < 0 || id >= c->vocab_size) die("--dump-logits: token id %ld is out of range\n", id);
            if (pos >= model.ctx) die("--dump-logits: more tokens than the context holds\n");
            float *logits = forward(&model, (int)id, pos++, 1);
            fwrite(logits, sizeof(float), (size_t)c->vocab_size, f);
            p = (*end == ',') ? end + 1 : end;
        }
        fclose(f);
        fprintf(stderr, "wrote logits for %d positions to %s\n", pos, o.dump_path);
        pool_shutdown();
        return 0;
    }

    /* Sampling defaults are the ones the Qwen3 model card recommends for each mode. */
    Chat ch;
    memset(&ch, 0, sizeof ch);
    ch.m = &model;
    ch.t = &tok;
    ch.o = &o;
    ch.s.temperature = o.temperature >= 0.0f ? o.temperature : (o.think ? 0.6f : 0.7f);
    ch.s.top_p = o.top_p >= 0.0f ? o.top_p : (o.think ? 0.95f : 0.8f);
    ch.s.top_k = o.top_k;
    ch.s.presence_penalty = o.presence_penalty;
    ch.s.rng = o.have_seed ? o.seed : (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32);
    ch.s.rng = hash_u64(ch.s.rng + 0x9E3779B97F4A7C15ULL);
    if (ch.s.rng == 0) ch.s.rng = 1; /* xorshift must not start at zero */
    ch.cand = xmalloc((size_t)c->vocab_size * sizeof(Cand));
    ch.seen = xcalloc((size_t)c->vocab_size, 1);
    ch.use_color = isatty(STDOUT_FILENO);
    if (o.log_path && !(ch.log = fopen(o.log_path, "w"))) die("cannot write %s\n", o.log_path);
    if (o.replay_path) { /* one scripted reply per line, token ids separated by spaces */
        char *text = read_file(o.replay_path, NULL);
        ch.replay = xcalloc(1, sizeof(IntVec));
        ch.n_replay = 1;
        for (char *p = text; *p;) {
            if (*p == '\n') {
                ch.replay = xrealloc(ch.replay, (size_t)(ch.n_replay + 1) * sizeof(IntVec));
                memset(&ch.replay[ch.n_replay++], 0, sizeof(IntVec));
                p++;
            } else if (*p >= '0' && *p <= '9') {
                iv_push(&ch.replay[ch.n_replay - 1], (int)strtol(p, &p, 10));
            } else {
                p++;
            }
        }
        free(text);
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sigint; /* Ctrl-C stops the current reply instead of killing the program */
    sigaction(SIGINT, &sa, NULL);

    if (o.raw) run_raw(&ch);
    else run_chat(&ch);
    if (ch.log) fclose(ch.log);

    pool_shutdown();
    return 0;
}
