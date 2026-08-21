#pragma once

#include "approxchol/graph.h"

#include <cstdint>
#include <random>
#include <vector>

namespace approxchol {

struct FactorEntry {
  idx row;
  double value;
};

struct FactorColumn {
  std::vector<FactorEntry> entries;
};

struct CholeskyFactor {
  std::vector<FactorColumn> columns;
  std::vector<idx> order;
};

CholeskyFactor factorize(const Graph &G, idx samples = 1,
                         bool exact_mode = false,
                         std::mt19937_64 *rng = nullptr);

CholeskyFactor ac1(const Graph &G, std::uint64_t seed = 42);
CholeskyFactor ac1(const Graph &G, std::mt19937_64 &rng);

CholeskyFactor ac2(const Graph &G, std::uint64_t seed = 42);
CholeskyFactor ac2(const Graph &G, std::mt19937_64 &rng);

CholeskyFactor exact(const Graph &G);

double get_entry(const CholeskyFactor &F, idx row, idx col);

} // namespace approxchol
