#include "kernel.h"

template <typename T, int VectorLength>
void VectorizationKernel<T, VectorLength>::Call(T* C, const T* A, const T* B,
                                                int M, int N, int K) {
  int iTile = 64, jTile = 64, kTile = 32;
  int iOuterBound = M / iTile, jOuterBound = N / jTile, kOuterBound = K / kTile;

  // mm128
  if (VectorLength == 128) {
    __m128 c, a, b;
    int floatNum = 128 / 32;

    for (int i_outer = 0; i_outer < iOuterBound; i_outer++) {
      for (int j_outer = 0; j_outer < jOuterBound; j_outer++) {
        for (int k_outer = 0; k_outer < kOuterBound; k_outer++) {
          for (int i_inner = 0; i_inner < iTile; i_inner++) {
            for (int k_inner = 0; k_inner < kTile; k_inner++) {
              a = _mm_set_ps1(A[(i_outer * iTile + i_inner) * K +
                                (k_outer * kTile + k_inner)]);
              for (int _j_inner = 0; _j_inner < jTile / floatNum; _j_inner++) {
                b = _mm_load_ps(B + (k_outer * kTile + k_inner) * N +
                                (j_outer * jTile + _j_inner * floatNum));
                c = _mm_mul_ps(b, a);
                a = _mm_load_ps(C + (i_outer * iTile + i_inner) * N +
                                (j_outer * jTile + _j_inner * floatNum));
                c = _mm_add_ps(c, a);
                _mm_storer_ps(C + (i_outer * iTile + i_inner) * N +
                                  (j_outer * jTile + _j_inner * floatNum),
                              c);
              }
            }
          }
        }
      }
    }
  }
  // mm256
  if (VectorLength == 256) {
    __m256 c, a, b;
    int floatNum = 256 / 32;

    for (int i_outer = 0; i_outer < iOuterBound; i_outer++) {
      for (int j_outer = 0; j_outer < jOuterBound; j_outer++) {
        for (int k_outer = 0; k_outer < kOuterBound; k_outer++) {
          for (int i_inner = 0; i_inner < iTile; i_inner++) {
            for (int k_inner = 0; k_inner < kTile; k_inner++) {
              a = _mm256_broadcast_ss(A + (i_outer * iTile + i_inner) * K +
                                      (k_outer * kTile + k_inner));
              for (int _j_inner = 0; _j_inner < jTile / floatNum; _j_inner++) {
                b = _mm256_load_ps(B + (k_outer * kTile + k_inner) * N +
                                   (j_outer * jTile + _j_inner * floatNum));
                c = _mm256_mul_ps(b, a);
                a = _mm256_load_ps(C + (i_outer * iTile + i_inner) * N +
                                   (j_outer * jTile + _j_inner * floatNum));
                c = _mm256_add_ps(c, a);
                _mm256_store_ps(C + (i_outer * iTile + i_inner) * N +
                                    (j_outer * jTile + _j_inner * floatNum),
                                c);
              }
            }
          }
        }
      }
    }
  }
}

// only float version is supportted, due to the limitation of sse.
template class AVX128VectorizationKernel<float>;
template class AVX256VectorizationKernel<float>;
