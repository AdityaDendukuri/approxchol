#pragma once

#include "approxchol/types.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace approxchol {

template <typename Index = std::size_t, typename Float = double>
struct DegreeQueue {
  static constexpr Index none = static_cast<Index>(-1);

  std::vector<Index> head;
  std::vector<Index> next;
  std::vector<Index> prev;
  std::vector<Index> degree;
  Index low = 0;

  explicit DegreeQueue(const Graph<Float, Index> &G) {
    const Index n = static_cast<Index>(G.size());
    head.assign(n + 1, none);
    next.assign(n, none);
    prev.assign(n, none);
    degree.assign(n, 0);

    for (Index i = 0; i < n; ++i) {
      Index d = 0;
      for (const auto &e : G[i]) {
        d += e.count;
      }
      insert(i, d);
    }
  }

  void insert(Index v, Index d) {
    if (d >= head.size()) {
      head.resize(std::max(d + 1, static_cast<Index>(2 * head.size())), none);
    }
    next[v] = head[d];
    prev[v] = none;
    if (head[d] != none) {
      prev[head[d]] = v;
    }
    head[d] = v;
    degree[v] = d;
    if (d < low) {
      low = d;
    }
  }

  void erase(Index v) {
    const Index d = degree[v];
    if (prev[v] == none) {
      head[d] = next[v];
    } else {
      next[prev[v]] = next[v];
    }
    if (next[v] != none) {
      prev[next[v]] = prev[v];
    }
    next[v] = none;
    prev[v] = none;
  }

  void rekey(Index v, Index d) {
    erase(v);
    insert(v, d);
  }

  Index pop_min() {
    while (low < head.size() && head[low] == none) {
      ++low;
    }
    if (low == head.size()) {
      throw std::runtime_error("DegreeQueue is empty");
    }
    const Index v = head[low];
    erase(v);
    return v;
  }
};

} // namespace approxchol
