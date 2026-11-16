/* cmp_f32.c -- compare two files of raw float32 values.
 *
 * usage: cmp_f32 A.bin B.bin TOLERANCE
 * Prints the largest absolute difference and exits non-zero if it exceeds the
 * tolerance (or if the files differ in length). Used by `./build.sh test` to check
 * the engine's logits against reference logits computed by `transformers`.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float *read_floats(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    *n = (size_t)size / sizeof(float);
    float *buf = malloc(*n * sizeof(float) + 1);
    if (!buf || fread(buf, sizeof(float), *n, f) != *n) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s A.bin B.bin TOLERANCE\n", argv[0]);
        return 2;
    }
    size_t na, nb;
    float *a = read_floats(argv[1], &na), *b = read_floats(argv[2], &nb);
    double tol = atof(argv[3]), max_diff = 0.0, max_val = 0.0;
    if (na != nb) {
        printf("FAIL: %zu values vs %zu values\n", na, nb);
        return 1;
    }
    for (size_t i = 0; i < na; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (!(d <= max_diff)) max_diff = d; /* written this way so a NaN is caught */
        if (fabs((double)b[i]) > max_val) max_val = fabs((double)b[i]);
    }
    int ok = max_diff <= tol;
    printf("%s: %zu values, largest difference %.3g (values range up to %.3g, tolerance %g)\n", ok ? "ok" : "FAIL", na, max_diff,
           max_val, tol);
    return ok ? 0 : 1;
}
