#pragma once

#include "approxchol/multigraph.h"

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

CholeskyFactor factorize(const MultiGraph &G, idx samples = 2,
                         bool exact_mode = false,
                         std::mt19937_64 *rng = nullptr);

CholeskyFactor ac1(const MultiGraph &G, std::uint64_t seed = 42);
CholeskyFactor ac1(const MultiGraph &G, std::mt19937_64 &rng);

CholeskyFactor ac2(const MultiGraph &G, std::uint64_t seed = 42);
CholeskyFactor ac2(const MultiGraph &G, std::mt19937_64 &rng);

CholeskyFactor exact(const MultiGraph &G);

double get_entry(const CholeskyFactor &F, idx row, idx col);

} // namespace approxchol
