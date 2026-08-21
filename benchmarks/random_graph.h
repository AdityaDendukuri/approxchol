#pragma once

#include "approxchol/graph.h"

#include <cstdint>
#include <random>
#include <stdexcept>

namespace approxchol {

inline Graph make_random_laplacian(idx n, idx avg_degree = 6,
                                   std::uint64_t seed = 42) {
  if (n < 2) {
    throw std::invalid_argument("n must be at least 2");
  }

  Graph G(n);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> w_dist(0.5, 2.0);

  for (idx i = 0; i < n; ++i) {
    idx j = (i + 1) % n;
    add_edge(G, i, j, w_dist(rng));
  }

  idx target_edges = (avg_degree * n) / 2;
  idx current_edges = n;

  std::uniform_int_distribution<idx> v_dist(0, n - 1);

  while (current_edges < target_edges) {
    idx u = v_dist(rng);
    idx v = v_dist(rng);
    if (u != v) {
      add_edge(G, u, v, w_dist(rng));
      ++current_edges;
    }
  }

  return G;
}

} // namespace approxchol
