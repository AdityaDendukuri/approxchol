#include "approxchol/matrix.h"
#include "approxchol/column_queue.h"

#include <algorithm>

namespace approxchol {

void add_connection(SparseMatrix &A, idx i, idx j, double weight, idx copies) {
  double split_val = weight / static_cast<double>(copies);
  for (idx c = 0; c < copies; ++c) {
    A[i].push_back({j, split_val});
    A[j].push_back({i, split_val});
  }
}

double extract_column(idx col, SparseMatrix &A,
                      const std::vector<std::uint8_t> &eliminated,
                      ColumnQueue &q, std::vector<MatrixEntry> &col_buf,
                      std::vector<ColumnNonzero> &nonzeros_out) {
  // Filter out entries in rows that are already eliminated
  col_buf.clear();
  for (const auto &entry : A[col]) {
    if (eliminated[entry.row]) {
      continue;
    }
    col_buf.push_back(entry);
    q.rekey(entry.row, q.nnz[entry.row] - 1);
  }

  A[col].clear();
  A[col].shrink_to_fit();

  if (col_buf.empty()) {
    return 0.0;
  }

  // Sort by row index to group parallel updates
  std::sort(
      col_buf.begin(), col_buf.end(),
      [](const MatrixEntry &a, const MatrixEntry &b) { return a.row < b.row; });

  // Linear merge pass
  nonzeros_out.clear();
  double diag_sum = 0.0;

  for (const auto &entry : col_buf) {
    if (!nonzeros_out.empty() && nonzeros_out.back().row == entry.row) {
      nonzeros_out.back().val += entry.val;
      ++nonzeros_out.back().budget;
    } else {
      nonzeros_out.push_back({entry.row, entry.val, 1});
    }
    diag_sum += entry.val;
  }

  return diag_sum;
}

void exact_outer_product(SparseMatrix &A, ColumnQueue &q,
                         const std::vector<ColumnNonzero> &nonzeros,
                         double diag_sum) {
  for (idx a = 0; a < nonzeros.size(); ++a) {
    for (idx b = a + 1; b < nonzeros.size(); ++b) {
      double w_exact = (nonzeros[a].val * nonzeros[b].val) / diag_sum;
      idx i = nonzeros[a].row;
      idx j = nonzeros[b].row;

      A[i].push_back({j, w_exact});
      A[j].push_back({i, w_exact});
      q.rekey(i, q.nnz[i] + 1);
      q.rekey(j, q.nnz[j] + 1);
    }
  }
}

void sample_outer_product(SparseMatrix &A, ColumnQueue &q,
                          std::vector<ColumnNonzero> &nonzeros, double diag_sum,
                          idx sample_limit, std::mt19937_64 &rng) {
  // Sort nonzeros by magnitude ascending
  std::sort(nonzeros.begin(), nonzeros.end(),
            [](const ColumnNonzero &a, const ColumnNonzero &b) {
              return a.val < b.val;
            });

  double tail_sum = diag_sum;

  for (idx a = 0; a + 1 < nonzeros.size(); ++a) {
    const auto &nz_a = nonzeros[a];
    tail_sum -= nz_a.val;

    if (tail_sum <= 0.0) {
      break;
    }

    const idx draws = std::min(nz_a.budget, sample_limit);
    const double val_bar = nz_a.val / static_cast<double>(draws);
    const double update_val = val_bar * (tail_sum / diag_sum);

    for (idx h = 0; h < draws; ++h) {
      std::uniform_real_distribution<double> dist(0.0, tail_sum);
      double target = dist(rng);

      idx b = a + 1;
      while (b + 1 < nonzeros.size() && target >= nonzeros[b].val) {
        target -= nonzeros[b].val;
        ++b;
      }

      idx i = nz_a.row;
      idx j = nonzeros[b].row;

      A[i].push_back({j, update_val});
      A[j].push_back({i, update_val});
      q.rekey(i, q.nnz[i] + 1);
      q.rekey(j, q.nnz[j] + 1);
    }
  }
}

} // namespace approxchol
