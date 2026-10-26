/// @file approxchol/types.hpp
/// @brief The graphs and factors that approximate Cholesky works with.
#pragma once

#include "approxchol/multigraph.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace approxchol {

/// @brief Default random source.
using rng64 = std::mt19937_64;

template <typename Float = double, std::integral Index = std::size_t>
using graph_edge = multi_edge<Float, Index>;

/// @brief A weighted multigraph as adjacency lists.
template <typename Float = double, std::integral Index = std::size_t>
using adjacency_list = std::vector<std::vector<graph_edge<Float, Index>>>;

template <typename Float = double, std::integral Index = std::size_t>
using neighbor = multi_edge<Float, Index>;

/// @brief One entry of a factor column.
template <typename Float = double, std::integral Index = std::size_t>
struct factor_entry {
    Index row{};
    Float value{};
};

/// @brief One column of a sparse Cholesky factor.
template <typename Float = double, std::integral Index = std::size_t>
struct factor_column {
    std::vector<factor_entry<Float, Index>> entries;
};

/// @brief A sparse Cholesky factor by columns, with its elimination order.
template <typename Float = double, std::integral Index = std::size_t>
struct cholesky_factor {
    std::vector<factor_column<Float, Index>> columns;
    std::vector<Index> order;
};

/// @brief Add an undirected edge of weight `weight` between `u` and `v`.
template <typename Float = double, std::integral Index = std::size_t>
inline void add_edge(adjacency_list<Float, Index> &G, Index u, Index v, Float weight) {
    G[u].push_back({v, weight, 1});
    G[v].push_back({u, weight, 1});
}

/// @brief The entry at (`row`, `col`) of a factor, or zero.
template <typename Float = double, std::integral Index = std::size_t>
inline Float get_entry(const cholesky_factor<Float, Index> &F, Index row, Index col) {
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
