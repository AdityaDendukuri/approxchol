/// @file approxchol/degree_queue.hpp  Bucket queue of vertices by degree (minimum degree).
#pragma once
#include <algorithm>
#include <concepts>
#include <vector>
namespace approxchol {
template <std::integral Index>
struct degree_queue {
    static constexpr Index none = static_cast<Index>(-1);
    std::vector<Index> head, next, prev, degree;
    Index low = 0;
    explicit degree_queue(Index n) : head(n + 1, none), next(n, none), prev(n, none), degree(n) {}
    void insert(Index v, Index d) {
        if (d >= head.size()) head.resize(std::max<std::size_t>(d + 1, 2 * head.size()), none);
        next[v] = head[d], prev[v] = none, degree[v] = d;
        if (head[d] != none) prev[head[d]] = v;
        head[d] = v, low = std::min(low, d);
    }
    void erase(Index v) {
        (prev[v] == none ? head[degree[v]] : next[prev[v]]) = next[v];
        if (next[v] != none) prev[next[v]] = prev[v];
    }
    void rekey(Index v, Index d) { erase(v), insert(v, d); }
    Index pop_min() {
        while (head[low] == none) ++low;
        const Index v = head[low]; return erase(v), v;
    }
};
} // namespace approxchol
