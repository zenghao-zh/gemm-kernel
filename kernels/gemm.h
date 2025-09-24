#pragma once

#include "common.h"


void matmul_w4a4_host(
        const Int4Storage *A,
        const Int4Storage *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);

void matmul_w4a8_host(
        const int8_t *A,
        const Int4Storage *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);


void matmul_w8a8_host(
        const int8_t *A,
        const int8_t *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);

void matmul_w4a8x4_host(
        const int8_t *A1,
        const int8_t *B1,
        const Int4Storage *A2,
        const Int4Storage *B2,
        uint32_t M1,
        uint32_t N1,
        uint32_t K1,
        uint32_t M2,
        uint32_t N2,
        uint32_t K2,
        int32_t *C
);