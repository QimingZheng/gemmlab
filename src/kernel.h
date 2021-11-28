#pragma once

#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string.h>

// openmp
#include "omp.h"

#include <mmintrin.h>   // mmx
#include <xmmintrin.h>  // sse
#include <emmintrin.h>  // sse2
#include <pmmintrin.h>  // sse3

#include <immintrin.h>

#include "metric.h"

using namespace std;

template <typename T>
class Kernel {
 public:
  Kernel() {}

  virtual void Call(T* C, const T* A, const T* B, int M, int N, int K) = 0;
};

template <typename T>
class NaiveKernel : public Kernel<T> {
 public:
  void Call(T* C, const T* A, const T* B, int M, int N, int K) override;
};

template <typename T>
class ReorderingKernel : public Kernel<T> {
 public:
  void Call(T* C, const T* A, const T* B, int M, int N, int K) override;
};

template <typename T>
class TilingKernel : public Kernel<T> {
 public:
  void Call(T* C, const T* A, const T* B, int M, int N, int K) override;
};

template <typename T, int VectorLength>
class VectorizationKernel : public Kernel<T> {
 public:
  void Call(T* C, const T* A, const T* B, int M, int N, int K) override;
};

template <typename T>
class AVX128VectorizationKernel : public VectorizationKernel<T, 128> {};

template <typename T>
class AVX256VectorizationKernel : public VectorizationKernel<T, 256> {};

template <typename T>
class ArrayPackingKernel : public Kernel<T> {
 public:
  ArrayPackingKernel() { packing = nullptr; }
  ~ArrayPackingKernel() {
    if (packing != nullptr) delete[] packing;
  }

  void Call(T* C, const T* A, const T* B, int M, int N, int K) override;

 private:
  T* packing;
};

template <typename T>
class WriteCacheKernel : public Kernel<T> {
 public:
  WriteCacheKernel() {
    packing = nullptr;
    wcache = nullptr;
  }
  ~WriteCacheKernel() {
    if (packing != nullptr) delete[] packing;
    if (wcache != nullptr) delete[] wcache;
  }

  void Call(T* C, const T* A, const T* B, int M, int N, int K) override;

 private:
  T* packing;
  T* wcache;
};

template <typename T>
class ParallelKernel : public Kernel<T> {
 public:
  ParallelKernel() {
    packing = nullptr;
    wcache = nullptr;
  }
  ~ParallelKernel() {
    if (packing != nullptr) delete[] packing;
    if (wcache != nullptr) delete[] wcache;
  }

  void Call(T* C, const T* A, const T* B, int M, int N, int K) override;

 private:
  T* packing;
  T* wcache;
};