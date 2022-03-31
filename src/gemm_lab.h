#pragma once

#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string.h>

#include "kernel.h"
#include "metric.h"

using namespace std;

template <typename T, class GemmKernel>
class Matrix {
 public:
  Matrix() = delete;
  Matrix(int R, int C) : R_(R), C_(C) {
    data_ = static_cast<T*>(aligned_alloc(8 * 4, R_ * C_ * sizeof(T)));
    memset(data_, 0, sizeof(T) * R_ * C_);
  }
  Matrix(const Matrix<T, GemmKernel>& other) {
    R_ = other.R_;
    C_ = other.C_;
    data_ = static_cast<T*>(aligned_alloc(8 * 4, R_ * C_ * sizeof(T)));
    memcpy(data_, other.data(), sizeof(T) * R_ * C_);
  }
  Matrix<T, GemmKernel>& operator=(const Matrix<T, GemmKernel>& other) {
    assert(R_, other.R_);
    assert(C_, other.C_);
    memcpy(data_, other.data_, sizeof(T) * R_ * C_);
  }

  // Move
  Matrix(Matrix<T, GemmKernel>&& other)
      : R_(other.R_), C_(other.C_), data_(other.data_) {
    other.R_ = 0;
    other.C_ = 0;
    other.data_ = nullptr;
  }
  Matrix<T, GemmKernel>& operator=(Matrix<T, GemmKernel>&& other) {
    assert(R_, other.R_);
    assert(C_, other.C_);
    std::swap(data_, other.data_);
  }

  Matrix<T, GemmKernel> operator*(const Matrix<T, GemmKernel>& other) {
    Matrix<T, GemmKernel> ret(R_, other.C_);
    GemmKernel kernel;
    // warm up
    kernel.Call(ret.data_, data_, other.data_, R_, other.C_, C_);
    // TODO: add a check here: (*this) == GroundTruth.

    // metrics collection
    {
      Metrics metrics(R_, other.C_, C_, loopCounter);

      for (int i = 0; i < loopCounter; i++) {
        kernel.Call(ret.data_, data_, other.data_, R_, other.C_, C_);
        // kernel.Call(ret.data_, data_, other.data_, R_, other.C_, C_);
      }
    }
    return ret;
  }

  bool operator==(const Matrix<T, GemmKernel>& other) {
    for (int i = 0; i < R_; i++) {
      for (int j = 0; j < C_; j++) {
        if (abs((*this)(i, j) - other(i, j)) > 1e-9) return false;
      }
    }
    return true;
  }

  T& operator()(int i, int j) { return data_[i * C_ + j]; }

  ~Matrix() { delete[] data_; }

  int row() { return R_; }
  int col() { return C_; }
  T* data() { return data_; }

 private:
  int loopCounter = 100;
  int R_, C_;
  T* data_ = nullptr;
};