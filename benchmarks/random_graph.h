#pragma once

#include "approxchol/approxchol.hpp"

#include <cstdint>
#include <random>
#include <stdexcept>

namespace approxchol
{

template <typename Float = double, typename Index = std::size_t>
inline Graph<Float, Index> make_random_laplacian(Index n, Index avg_degree = 6,
                                                 std::uint64_t seed = 42)
{
    if (n < 2) {
        throw std::invalid_argument("n must be at least 2");
    }

    Graph<Float, Index> G(n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Float> w_dist(static_cast<Float>(0.5), static_cast<Float>(2.0));

    for (Index i = 0; i < n; ++i) {
        Index j = (i + 1) % n;
        add_edge(G, i, j, w_dist(rng));
    }

    Index target_edges = (avg_degree * n) / 2;
    Index current_edges = n;

    std::uniform_int_distribution<Index> v_dist(0, n - 1);

    while (current_edges < target_edges) {
        Index u = v_dist(rng);
        Index v = v_dist(rng);
        if (u != v) {
            add_edge(G, u, v, w_dist(rng));
            ++current_edges;
        }
    }

    return G;
}

} // namespace approxchol
