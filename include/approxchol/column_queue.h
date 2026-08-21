#pragma once

#include "approxchol/matrix.h"

#include <vector>

namespace approxchol {

struct ColumnQueue {
  static constexpr idx none = static_cast<idx>(-1);

  std::vector<idx> head;
  std::vector<idx> next;
  std::vector<idx> prev;
  std::vector<idx> nnz;
  idx low = 0;

  explicit ColumnQueue(const SparseMatrix &A);

  void insert(idx col, idx count);
  void erase(idx col);
  void rekey(idx col, idx count);
  idx pop_min();
};

} // namespace approxchol
