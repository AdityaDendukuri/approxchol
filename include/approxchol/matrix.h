#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace approxchol {

using idx = std::size_t;

struct ColumnQueue;

struct MatrixEntry {
  idx row;
  double val;
};

using SparseMatrix = std::vector<std::vector<MatrixEntry>>;

struct ColumnNonzero {
  idx row;
  double val;
  idx budget;
};

void add_connection(SparseMatrix &A, idx i, idx j, double weight,
                    idx copies = 1);

double extract_column(idx col, SparseMatrix &A,
                      const std::vector<std::uint8_t> &eliminated,
                      ColumnQueue &q, std::vector<MatrixEntry> &col_buf,
                      std::vector<ColumnNonzero> &nonzeros_out);

void exact_outer_product(SparseMatrix &A, ColumnQueue &q,
                         const std::vector<ColumnNonzero> &nonzeros,
                         double diag_sum);

void sample_outer_product(SparseMatrix &A, ColumnQueue &q,
                          std::vector<ColumnNonzero> &nonzeros, double diag_sum,
                          idx sample_limit, std::mt19937_64 &rng);

} // namespace approxchol
