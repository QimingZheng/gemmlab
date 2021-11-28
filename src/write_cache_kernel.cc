#include "kernel.h"

template <typename T>
void WriteCacheKernel<T>::Call(T* C, const T* A, const T* B, int M, int N,
                               int K) {
  int iTile = 64, jTile = 64, kTile = 32;
  int iOuterBound = M / iTile, jOuterBound = N / jTile, kOuterBound = K / kTile;

  __m256 c, a, b, d, e;
  int floatNum = 256 / 32;

  if (packing == nullptr) {
    packing = static_cast<T*>(aligned_alloc(8 * 4, K * N * sizeof(T)));
  }

  if (wcache == nullptr) {
    wcache = static_cast<T*>(aligned_alloc(8 * 4, iTile * jTile * sizeof(T)));
  }

  for (int k_outer = 0; k_outer < kOuterBound; k_outer++) {
    for (int j_outer = 0; j_outer < jOuterBound; j_outer++) {
      for (int k_inner = 0; k_inner < kTile; k_inner++) {
        for (int _j_inner = 0; _j_inner < jTile / floatNum; _j_inner++) {
          b = _mm256_load_ps(B + (k_outer * kTile + k_inner) * N +
                             (j_outer * jTile + _j_inner * floatNum));
          _mm256_store_ps(packing + j_outer * K * jTile +
                              k_outer * jTile * kTile + k_inner * jTile +
                              _j_inner * floatNum,
                          b);
        }
      }
    }
  }

  for (int i_outer = 0; i_outer < iOuterBound; i_outer++) {
    for (int j_outer = 0; j_outer < jOuterBound; j_outer++) {
      memset(wcache, 0, sizeof(T) * iTile * jTile);
      for (int k_outer = 0; k_outer < kOuterBound; k_outer++) {
        for (int i_inner = 0; i_inner < iTile; i_inner++) {
          for (int k_inner = 0; k_inner < kTile; k_inner++) {
            a = _mm256_broadcast_ss(A + (i_outer * iTile + i_inner) * K +
                                    (k_outer * kTile + k_inner));
            for (int _j_inner = 0; _j_inner < jTile / floatNum; _j_inner++) {
              b = _mm256_load_ps(packing + j_outer * K * jTile +
                                 k_outer * jTile * kTile + k_inner * jTile +
                                 _j_inner * floatNum);
              c = _mm256_mul_ps(b, a);
              d = _mm256_load_ps(wcache + i_inner * jTile +
                                 _j_inner * floatNum);
              e = _mm256_add_ps(c, d);
              _mm256_store_ps(wcache + i_inner * jTile + _j_inner * floatNum,
                              e);
            }
          }
        }
      }
      // write back
      for (int i_inner = 0; i_inner < iTile; i_inner++) {
        memcpy(C + (i_outer * iTile + i_inner) * N + j_outer * jTile,
               wcache + i_inner * jTile, jTile * sizeof(T));
      }
    }
  }
}

template class WriteCacheKernel<float>;
