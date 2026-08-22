#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace approxchol
{

using idx = std::size_t;

template <typename Float = double, typename Index = std::size_t> struct Edge {
    Index to;
    Float weight;
    std::uint8_t count = 1;
};

template <typename Float = double, typename Index = std::size_t>
using Graph = std::vector<std::vector<Edge<Float, Index>>>;

template <typename Float = double, typename Index = std::size_t> struct Neighbor {
    Index to;
    Float weight;
    std::uint8_t count = 1;
};

template <typename Float = double, typename Index = std::size_t> struct FactorEntry {
    Index row;
    Float value;
};

template <typename Float = double, typename Index = std::size_t> struct FactorColumn {
    std::vector<FactorEntry<Float, Index>> entries;
};

template <typename Float = double, typename Index = std::size_t> struct CholeskyFactor {
    std::vector<FactorColumn<Float, Index>> columns;
    std::vector<Index> order;
};

template <typename Float = double, typename Index = std::size_t>
inline void add_edge(Graph<Float, Index> &G, Index u, Index v, Float weight)
{
    G[u].push_back({v, weight, 1});
    G[v].push_back({u, weight, 1});
}

template <typename Float = double, typename Index = std::size_t>
inline Float get_entry(const CholeskyFactor<Float, Index> &F, Index row, Index col)
{
    if (col >= F.columns.size()) {
        return static_cast<Float>(0);
    }
    for (const auto &entry : F.columns[col].entries) {
        if (entry.row == row) {
            return entry.value;
        }
    }
    return static_cast<Float>(0);
}

} // namespace approxchol
