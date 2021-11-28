#include "kernel.h"

template <typename T>
void TilingKernel<T>::Call(T* C, const T* A, const T* B, int M, int N, int K) {
  int iTile = 64, jTile = 64, kTile = 32;
  int iOuterBound = M / iTile, jOuterBound = N / jTile, kOuterBound = K / kTile;
  for (int i_outer = 0; i_outer < iOuterBound; i_outer++) {
    for (int j_outer = 0; j_outer < jOuterBound; j_outer++) {
      for (int k_outer = 0; k_outer < kOuterBound; k_outer++) {
        for (int i_inner = 0; i_inner < iTile; i_inner++) {
          for (int k_inner = 0; k_inner < kTile; k_inner++) {
            for (int j_inner = 0; j_inner < jTile; j_inner++) {
              C[(i_outer * iTile + i_inner) * N +
                (j_outer * jTile + j_inner)] +=
                  A[(i_outer * iTile + i_inner) * K +
                    (k_outer * kTile + k_inner)] *
                  B[(k_outer * kTile + k_inner) * N +
                    (j_outer * jTile + j_inner)];
            }
          }
        }
      }
    }
  }
}

template class TilingKernel<int>;
template class TilingKernel<double>;
template class TilingKernel<float>;
