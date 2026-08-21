#pragma once

#include "approxchol/graph.h"

#include <vector>

namespace approxchol {

struct DegreeQueue {
  static constexpr idx none = static_cast<idx>(-1);

  std::vector<idx> head;
  std::vector<idx> next;
  std::vector<idx> prev;
  std::vector<idx> degree;
  idx low = 0;

  explicit DegreeQueue(const Graph &G);

  void insert(idx v, idx d);
  void erase(idx v);
  void rekey(idx v, idx d);
  idx pop_min();
};

} // namespace approxchol
