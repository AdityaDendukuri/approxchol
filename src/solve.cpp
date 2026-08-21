#include "approxchol/solve.h"
#include "approxchol/degree_queue.h"

#include <cmath>
#include <cstdint>

namespace approxchol {

namespace {

void write_column(idx v, idx step, double total_weight,
                  const std::vector<Neighbor> &nbr, CholeskyFactor &factor) {
  const double sqrt_total = std::sqrt(total_weight);

  auto &col_entries = factor.columns[step].entries;
  col_entries.reserve(nbr.size() + 1);
  col_entries.push_back({v, sqrt_total});
  for (const auto &n_info : nbr) {
    col_entries.push_back({n_info.to, -n_info.weight / sqrt_total});
  }
}

void permute_rows(CholeskyFactor &factor) {
  const idx n = factor.order.size();
  std::vector<idx> position(n);

  for (idx step = 0; step < n; ++step) {
    position[factor.order[step]] = step;
  }

  for (idx step = 0; step < n; ++step) {
    for (auto &entry : factor.columns[step].entries) {
      entry.row = position[entry.row];
    }
  }
}

} // namespace

CholeskyFactor factorize(const Graph &input_G, idx samples,
                         bool exact_mode, std::mt19937_64 *rng) {
  const idx n = input_G.size();
  Graph G = input_G;

  if (samples > 1 && !exact_mode) {
    for (auto &row : G) {
      for (auto &e : row) {
        e.count = static_cast<std::uint8_t>(samples);
      }
    }
  }

  DegreeQueue q(G);

  std::vector<std::uint8_t> done(n, 0);

  CholeskyFactor factor;
  factor.columns.resize(n);
  factor.order.reserve(n);

  std::vector<Edge> star_buf;
  star_buf.reserve(128);

  std::vector<Neighbor> nbr_buf;
  nbr_buf.reserve(128);

  std::mt19937_64 local_rng(42);
  std::mt19937_64 &active_rng = rng ? *rng : local_rng;

  for (idx step = 0; step + 1 < n; ++step) {
    const idx v = q.pop_min();
    factor.order.push_back(v);

    double total_weight =
        collect_neighbors(v, G, done, q, star_buf, nbr_buf);
    done[v] = 1;

    if (total_weight <= 0.0) {
      continue;
    }

    write_column(v, step, total_weight, nbr_buf, factor);

    if (exact_mode) {
      add_exact_clique(G, q, nbr_buf, total_weight);
    } else {
      sample_clique(G, q, nbr_buf, total_weight, samples, active_rng);
    }
  }

  // Nullspace vertex
  const idx last_v = q.pop_min();
  factor.order.push_back(last_v);

  permute_rows(factor);

  return factor;
}

CholeskyFactor ac1(const Graph &G, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  return factorize(G, 1, false, &rng);
}

CholeskyFactor ac1(const Graph &G, std::mt19937_64 &rng) {
  return factorize(G, 1, false, &rng);
}

CholeskyFactor ac2(const Graph &G, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  return factorize(G, 2, false, &rng);
}

CholeskyFactor ac2(const Graph &G, std::mt19937_64 &rng) {
  return factorize(G, 2, false, &rng);
}

CholeskyFactor exact(const Graph &G) {
  return factorize(G, 1, true, nullptr);
}

double get_entry(const CholeskyFactor &F, idx row, idx col) {
  if (col >= F.columns.size()) {
    return 0.0;
  }
  for (const auto &entry : F.columns[col].entries) {
    if (entry.row == row) {
      return entry.value;
    }
  }
  return 0.0;
}

} // namespace approxchol
