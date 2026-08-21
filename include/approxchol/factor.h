#pragma once

#include "approxchol/multigraph.h"

#include <cstddef>
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

double get_entry(const CholeskyFactor &F, idx row, idx col);

namespace detail {
void write_column(idx v, idx step, double total_weight,
                  const std::vector<Neighbor> &nbr, CholeskyFactor &factor);
void permute_rows(CholeskyFactor &factor);
} // namespace detail

} // namespace approxchol
