#pragma once

#include "approxchol/matrix.h"

#include <cstdint>
#include <random>
#include <stdexcept>

namespace approxchol {

inline SparseMatrix make_random_laplacian(idx n, idx avg_nonzeros = 6,
                                         idx k_split = 2,
                                         std::uint64_t seed = 42) {
  if (n < 2) {
    throw std::invalid_argument("n must be at least 2");
  }

  SparseMatrix A(n);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> w_dist(0.5, 2.0);

  for (idx i = 0; i < n; ++i) {
    idx j = (i + 1) % n;
    add_connection(A, i, j, w_dist(rng), k_split);
  }

  idx target_connections = (avg_nonzeros * n) / 2;
  idx current_connections = n;

  std::uniform_int_distribution<idx> v_dist(0, n - 1);

  while (current_connections < target_connections) {
    idx u = v_dist(rng);
    idx v = v_dist(rng);
    if (u != v) {
      add_connection(A, u, v, w_dist(rng), k_split);
      ++current_connections;
    }
  }

  return A;
}

} // namespace approxchol
