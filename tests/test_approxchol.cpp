#include "approxchol/degree_queue.h"
#include "approxchol/graph.h"
#include "approxchol/solve.h"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using approxchol::idx;

void check(bool condition, const char *msg) {
  if (!condition) {
    throw std::runtime_error(std::string("Test assertion failed: ") + msg);
  }
}

approxchol::Graph make_grid_laplacian(idx rows, idx cols) {
  const idx n = rows * cols;
  approxchol::Graph G(n);

  auto index = [cols](idx r, idx c) { return r * cols + c; };

  for (idx r = 0; r < rows; ++r) {
    for (idx c = 0; c < cols; ++c) {
      idx u = index(r, c);
      if (c + 1 < cols) {
        approxchol::add_edge(G, u, index(r, c + 1), 1.0);
      }
      if (r + 1 < rows) {
        approxchol::add_edge(G, u, index(r + 1, c), 1.0);
      }
    }
  }

  return G;
}

std::vector<std::vector<double>>
dense_from_graph(const approxchol::Graph &G) {
  const idx n = G.size();
  std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));

  for (idx u = 0; u < n; ++u) {
    double diag = 0.0;
    for (const auto &e : G[u]) {
      A[u][e.to] -= e.weight;
      diag += e.weight;
    }
    A[u][u] = diag;
  }

  return A;
}

void test_exact_factorization() {
  const idx rows = 4;
  const idx cols = 4;
  const idx n = rows * cols;

  approxchol::Graph G = make_grid_laplacian(rows, cols);
  auto A_dense = dense_from_graph(G);

  approxchol::CholeskyFactor F = approxchol::exact(G);

  std::vector<idx> pos(n);
  for (idx step = 0; step < n; ++step) {
    pos[F.order[step]] = step;
  }

  double max_err = 0.0;
  for (idx u = 0; u < n; ++u) {
    for (idx v = 0; v < n; ++v) {
      double l_lt = 0.0;
      for (idx k = 0; k < n; ++k) {
        double l_uk = approxchol::get_entry(F, pos[u], k);
        double l_vk = approxchol::get_entry(F, pos[v], k);
        if (l_uk != 0.0 && l_vk != 0.0) {
          l_lt += l_uk * l_vk;
        }
      }
      max_err = std::max(max_err, std::abs(l_lt - A_dense[u][v]));
    }
  }

  check(max_err < 1e-12, "Exact Cholesky error exceeds tolerance");
  std::cout << "test_exact_factorization passed (max error = " << max_err
            << ")\n";
}

void test_ac1_and_ac2_factorization() {
  const idx rows = 4;
  const idx cols = 4;
  const idx n = rows * cols;

  approxchol::Graph G = make_grid_laplacian(rows, cols);

  approxchol::CholeskyFactor F1 = approxchol::ac1(G, /*seed=*/42);
  approxchol::CholeskyFactor F2 = approxchol::ac2(G, /*seed=*/42);

  check(F1.columns.size() == n, "AC1 factor column count mismatch");
  check(F2.columns.size() == n, "AC2 factor column count mismatch");

  std::cout << "test_ac1_and_ac2_factorization passed\n";
}

void test_full_factor_expectation() {
  const idx rows = 3;
  const idx cols = 3;
  const idx n = rows * cols;
  const idx trials = 3000;

  approxchol::Graph G = make_grid_laplacian(rows, cols);
  auto A_dense = dense_from_graph(G);

  std::vector<std::vector<double>> E_LLt(n, std::vector<double>(n, 0.0));
  std::mt19937_64 rng(123);

  for (idx t = 0; t < trials; ++t) {
    approxchol::CholeskyFactor F = approxchol::ac2(G, rng);

    std::vector<idx> pos(n);
    for (idx step = 0; step < n; ++step) {
      pos[F.order[step]] = step;
    }

    for (idx u = 0; u < n; ++u) {
      idx row_u = pos[u];
      for (idx v = 0; v < n; ++v) {
        idx row_v = pos[v];
        double val = 0.0;
        for (idx k = 0; k < n; ++k) {
          double l_uk = approxchol::get_entry(F, row_u, k);
          double l_vk = approxchol::get_entry(F, row_v, k);
          if (l_uk != 0.0 && l_vk != 0.0) {
            val += l_uk * l_vk;
          }
        }
        E_LLt[u][v] += val;
      }
    }
  }

  double max_mc_err = 0.0;
  for (idx u = 0; u < n; ++u) {
    for (idx v = 0; v < n; ++v) {
      double mean_val = E_LLt[u][v] / static_cast<double>(trials);
      max_mc_err = std::max(max_mc_err, std::abs(mean_val - A_dense[u][v]));
    }
  }

  check(max_mc_err < 0.1, "Full factor expectation error too large");
  std::cout << "test_full_factor_expectation passed (max error = " << max_mc_err
            << ")\n";
}

} // namespace

int main() {
  try {
    test_exact_factorization();
    test_ac1_and_ac2_factorization();
    test_full_factor_expectation();
    std::cout << "All tests passed.\n";
  } catch (const std::exception &e) {
    std::cerr << "Test failed: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
