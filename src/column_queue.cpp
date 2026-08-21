#include "approxchol/column_queue.h"

#include <algorithm>
#include <stdexcept>

namespace approxchol {

ColumnQueue::ColumnQueue(const SparseMatrix &A) {
  const idx n = A.size();
  head.assign(n + 1, none);
  next.assign(n, none);
  prev.assign(n, none);
  nnz.assign(n, 0);

  for (idx i = 0; i < n; ++i) {
    insert(i, A[i].size());
  }
}

void ColumnQueue::insert(idx col, idx count) {
  if (count >= head.size()) {
    head.resize(std::max(count + 1, 2 * head.size()), none);
  }
  next[col] = head[count];
  prev[col] = none;
  if (head[count] != none) {
    prev[head[count]] = col;
  }
  head[count] = col;
  nnz[col] = count;
  if (count < low) {
    low = count;
  }
}

void ColumnQueue::erase(idx col) {
  const idx count = nnz[col];
  if (prev[col] == none) {
    head[count] = next[col];
  } else {
    next[prev[col]] = next[col];
  }
  if (next[col] != none) {
    prev[next[col]] = prev[col];
  }
  next[col] = none;
  prev[col] = none;
}

void ColumnQueue::rekey(idx col, idx count) {
  erase(col);
  insert(col, count);
}

idx ColumnQueue::pop_min() {
  while (low < head.size() && head[low] == none) {
    ++low;
  }
  if (low == head.size()) {
    throw std::runtime_error("ColumnQueue is empty");
  }
  const idx col = head[low];
  erase(col);
  return col;
}

} // namespace approxchol
