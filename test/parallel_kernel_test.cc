#include "src/gemm_lab.h"

using namespace std;

int main() {
  int M = 4096, K = 4096, N = 4096;
  Matrix<float, ParallelKernel<float>> a(M, K);
  Matrix<float, ParallelKernel<float>> b(K, N);
  auto c = a * b;
  return 0;
}