#include "src/gemm_lab.h"

using namespace std;

int main() {
  int M = 1024, K = 1024, N = 1024;
  Matrix<float, TilingKernel<float>> a(M, K);
  Matrix<float, TilingKernel<float>> b(K, N);
  auto c = a * b;
  return 0;
}