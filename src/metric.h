#pragma once

#include <sys/time.h>
#include <iostream>

class Timer {
 public:
  Timer() = delete;
  Timer(int64_t &uselapsed) : uselapsed(uselapsed) {
    gettimeofday(&start, nullptr);
  }

  ~Timer() {
    gettimeofday(&end, nullptr);
    uselapsed = 1000L * 1000L * (end.tv_sec - start.tv_sec) +
                (end.tv_usec - start.tv_usec);
  }

 private:
  timeval start, end;
  int64_t &uselapsed;
};

class Metrics {
 public:
  Metrics(int M, int K, int N, int loop) : M(M), N(N), K(K), loop(loop) {
    timer = new Timer(uselapsed);
  }
  ~Metrics() {
    delete timer;
    std::cout << "[M, K, N] = [" << M << "," << K << "," << N << "], elapsed "
              << uselapsed / loop << "us / " << uselapsed / (1000L * loop)
              << "ms\n";
    std::cout << "Performance = " << (2L * M * N * K * loop) / (uselapsed)
              << " MFlops, " << (2L * M * N * K * loop) / (1000L * uselapsed)
              << "GFplos\n";
  }

 private:
  int loop;
  int M, N, K;
  Timer *timer;
  int64_t uselapsed = 0;
};