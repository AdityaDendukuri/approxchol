#pragma once

#include "approxchol/matrix.h"

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

CholeskyFactor factorize(const SparseMatrix &A, idx samples = 2,
                         bool exact_mode = false,
                         std::mt19937_64 *rng = nullptr);

CholeskyFactor ac1(const SparseMatrix &A, std::uint64_t seed = 42);
CholeskyFactor ac1(const SparseMatrix &A, std::mt19937_64 &rng);

CholeskyFactor ac2(const SparseMatrix &A, std::uint64_t seed = 42);
CholeskyFactor ac2(const SparseMatrix &A, std::mt19937_64 &rng);

CholeskyFactor exact(const SparseMatrix &A);

double get_entry(const CholeskyFactor &F, idx row, idx col);

} // namespace approxchol
