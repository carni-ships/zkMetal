/*
 * example.c -- Minimal zkMetal usage example
 *
 * Compile (static, from bindings/c/):
 *   cc -O2 -I include example.c build/libzkmetal.a -o example
 *
 * Or with pkg-config (after install):
 *   cc -O2 $(pkg-config --cflags zkmetal) example.c $(pkg-config --libs zkmetal) -o example
 */

#include <stdio.h>
#include <string.h>
#include "zkmetal.h"

static void print_fr(const char *label, const uint64_t v[4]) {
    printf("%s: [0x%016llx, 0x%016llx, 0x%016llx, 0x%016llx]\n",
           label, v[0], v[1], v[2], v[3]);
}

int main(void) {
    printf("zkMetal v%d.%d.%d example\n\n",
           ZKMETAL_VERSION_MAJOR, ZKMETAL_VERSION_MINOR, ZKMETAL_VERSION_PATCH);

    /* ---- BN254 Fr arithmetic ---- */
    printf("=== BN254 Fr Arithmetic ===\n");

    /* R^2 mod p (Montgomery form of 1 is R mod p) */
    uint64_t a[4] = {0x0000000000000001, 0x0000000000000000,
                     0x0000000000000000, 0x0000000000000000};
    uint64_t b[4] = {0x0000000000000002, 0x0000000000000000,
                     0x0000000000000000, 0x0000000000000000};
    uint64_t r[4];

    /* Convert to Montgomery form: mont(x) = x * R mod p */
    /* For this example we just multiply raw values */
    bn254_fr_mul(a, b, r);
    print_fr("mul(1, 2)", r);

    bn254_fr_add(a, b, r);
    print_fr("add(1, 2)", r);

    bn254_fr_sub(b, a, r);
    print_fr("sub(2, 1)", r);

    /* ---- Keccak-256 ---- */
    printf("\n=== Keccak-256 ===\n");

    const uint8_t msg[] = "zkMetal";
    uint8_t hash[32];
    keccak256_hash_neon(msg, sizeof(msg) - 1, hash);

    printf("keccak256(\"zkMetal\"): ");
    for (int i = 0; i < 32; i++) printf("%02x", hash[i]);
    printf("\n");

    /* ---- Blake3 ---- */
    printf("\n=== Blake3 ===\n");

    blake3_hash_neon(msg, sizeof(msg) - 1, hash);
    printf("blake3(\"zkMetal\"):    ");
    for (int i = 0; i < 32; i++) printf("%02x", hash[i]);
    printf("\n");

    /* ---- Poseidon2 ---- */
    printf("\n=== Poseidon2 (BN254 Fr) ===\n");

    uint64_t pa[4] = {1, 0, 0, 0};
    uint64_t pb[4] = {2, 0, 0, 0};
    uint64_t ph[4];
    poseidon2_hash_cpu(pa, pb, ph);
    print_fr("poseidon2(1, 2)", ph);

    /* ---- BN254 Pairing ---- */
    printf("\n=== BN254 Pairing Check ===\n");

    /* A real pairing check requires valid G1/G2 points on-curve.
       This is just a demonstration of the API call pattern.
       In practice you would load real SRS points here. */
    printf("(skipped -- requires valid curve points)\n");

    printf("\nDone.\n");
    return 0;
}
