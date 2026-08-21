#include "approxchol/solve.h"
#include "approxchol/column_queue.h"

#include <cmath>
#include <cstdint>

namespace approxchol {

namespace {

void write_column(idx col, idx step, double diag_sum,
                  const std::vector<ColumnNonzero> &nonzeros,
                  CholeskyFactor &factor) {
  const double sqrt_diag = std::sqrt(diag_sum);

  factor.columns[step].entries.push_back({col, sqrt_diag});
  for (const auto &nz : nonzeros) {
    factor.columns[step].entries.push_back({nz.row, -nz.val / sqrt_diag});
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

CholeskyFactor factorize(const SparseMatrix &input_A, idx samples,
                         bool exact_mode, std::mt19937_64 *rng) {
  const idx n = input_A.size();
  SparseMatrix A = input_A;
  ColumnQueue q(A);

  std::vector<std::uint8_t> eliminated(n, 0);

  CholeskyFactor factor;
  factor.columns.resize(n);
  factor.order.reserve(n);

  std::vector<MatrixEntry> col_buf;
  std::vector<ColumnNonzero> nonzeros_buf;

  std::mt19937_64 local_rng(42);
  std::mt19937_64 &active_rng = rng ? *rng : local_rng;

  for (idx step = 0; step + 1 < n; ++step) {
    const idx col = q.pop_min();
    factor.order.push_back(col);

    double diag_sum =
        extract_column(col, A, eliminated, q, col_buf, nonzeros_buf);
    eliminated[col] = 1;

    if (diag_sum <= 0.0) {
      continue;
    }

    write_column(col, step, diag_sum, nonzeros_buf, factor);

    if (exact_mode) {
      exact_outer_product(A, q, nonzeros_buf, diag_sum);
    } else {
      sample_outer_product(A, q, nonzeros_buf, diag_sum, samples, active_rng);
    }
  }

  // Nullspace column
  const idx last_col = q.pop_min();
  factor.order.push_back(last_col);

  permute_rows(factor);

  return factor;
}

CholeskyFactor ac1(const SparseMatrix &A, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  return factorize(A, 1, false, &rng);
}

CholeskyFactor ac1(const SparseMatrix &A, std::mt19937_64 &rng) {
  return factorize(A, 1, false, &rng);
}

CholeskyFactor ac2(const SparseMatrix &A, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  return factorize(A, 2, false, &rng);
}

CholeskyFactor ac2(const SparseMatrix &A, std::mt19937_64 &rng) {
  return factorize(A, 2, false, &rng);
}

CholeskyFactor exact(const SparseMatrix &A) {
  return factorize(A, 1, true, nullptr);
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
