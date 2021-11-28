#include "kernel.h"

template <typename T>
void NaiveKernel<T>::Call(T* C, const T* A, const T* B, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T ma = 0;
      for (int k = 0; k < K; k++) {
        ma += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = ma;
    }
  }
}

template class NaiveKernel<int>;
template class NaiveKernel<double>;
template class NaiveKernel<float>;
