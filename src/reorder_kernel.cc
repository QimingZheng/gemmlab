#include "kernel.h"

template <typename T>
void ReorderingKernel<T>::Call(T* C, const T* A, const T* B, int M, int N,
                               int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

template class ReorderingKernel<int>;
template class ReorderingKernel<double>;
template class ReorderingKernel<float>;
