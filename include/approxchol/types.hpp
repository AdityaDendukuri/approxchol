/// @file approxchol/types.hpp  The graph being eliminated and the factor it produces.
#pragma once
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
namespace approxchol {
using rng64 = std::mt19937_64;
/// An edge to `to` of weight `weight`, standing for `count` parallel copies.
template <typename Float = double, std::integral Index = std::size_t>
struct edge { Index to{}; Float weight{1}; std::uint8_t count = 1; };

/// One neighbor list per vertex; each undirected edge is stored in both lists.
template <typename Float = double, std::integral Index = std::size_t>
using adjacency_list = std::vector<std::vector<edge<Float, Index>>>;

template <typename Float, std::integral Index>
void add_edge(adjacency_list<Float, Index> &G, Index u, Index v, Float w) {
    G[u].push_back({v, w, 1}), G[v].push_back({u, w, 1});
}

/// Lower factor by columns, rows in elimination positions: P L_G P^T ~ L L^T,
/// where `order[k]` is the vertex eliminated at step k.
template <typename Float = double, std::integral Index = std::size_t>
struct cholesky_factor {
    struct entry { Index row{}; Float value{}; };
    std::vector<std::vector<entry>> columns;
    std::vector<Index> order;
};
} // namespace approxchol
