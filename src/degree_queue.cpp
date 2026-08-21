#include "approxchol/degree_queue.h"

#include <algorithm>
#include <stdexcept>

namespace approxchol {

DegreeQueue::DegreeQueue(const MultiGraph &G) {
  const idx n = G.size();
  head.assign(n + 1, none);
  next.assign(n, none);
  prev.assign(n, none);
  degree.assign(n, 0);

  for (idx i = 0; i < n; ++i) {
    insert(i, G[i].size());
  }
}

void DegreeQueue::insert(idx v, idx d) {
  if (d >= head.size()) {
    head.resize(std::max(d + 1, 2 * head.size()), none);
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

void DegreeQueue::erase(idx v) {
  const idx d = degree[v];
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

void DegreeQueue::rekey(idx v, idx d) {
  erase(v);
  insert(v, d);
}

idx DegreeQueue::pop_min() {
  while (low < head.size() && head[low] == none) {
    ++low;
  }
  if (low == head.size()) {
    throw std::runtime_error("DegreeQueue is empty");
  }
  const idx v = head[low];
  erase(v);
  return v;
}

} // namespace approxchol
