#pragma once

#include "approxchol/clique.hpp"
#include "approxchol/degree_queue.hpp"
#include "approxchol/types.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace approxchol {

namespace detail {

template <typename Float, typename Index>
inline void write_column(Index v, Index step, Float total_weight,
                         const std::vector<Neighbor<Float, Index>> &nbr,
                         CholeskyFactor<Float, Index> &factor) {
    const Float sqrt_total = std::sqrt(total_weight);

    auto &col_entries = factor.columns[step].entries;
    col_entries.reserve(nbr.size() + 1);
    col_entries.push_back({v, sqrt_total});
    for (const auto &n_info : nbr) {
        col_entries.push_back({n_info.to, -n_info.weight / sqrt_total});
    }
}

template <typename Float, typename Index>
inline void permute_rows(CholeskyFactor<Float, Index> &factor) {
    const Index n = static_cast<Index>(factor.order.size());
    std::vector<Index> position(n);

    for (Index step = 0; step < n; ++step) {
        position[factor.order[step]] = step;
    }

    for (Index step = 0; step < n; ++step) {
        for (auto &entry : factor.columns[step].entries) {
            entry.row = position[entry.row];
        }
    }
}

} // namespace detail

template <typename Float = double, typename Index = std::size_t, typename Rng = std::mt19937_64,
          typename Queue = DegreeQueue<Index, Float>>
inline CholeskyFactor<Float, Index> factorize(const Graph<Float, Index> &input_G, Index samples = 1,
                                              bool exact_mode = false, Rng *rng = nullptr) {
    const Index n = static_cast<Index>(input_G.size());
    Graph<Float, Index> G = input_G;

    if (samples > 1 && !exact_mode) {
        for (auto &row : G) {
            for (auto &e : row) {
                e.count = static_cast<std::uint8_t>(samples);
            }
        }
    }

    Queue q(G);
    std::vector<std::uint8_t> done(n, 0);

    CholeskyFactor<Float, Index> factor;
    factor.columns.resize(n);
    factor.order.reserve(n);

    std::vector<Edge<Float, Index>> star_buf;
    star_buf.reserve(128);

    std::vector<Neighbor<Float, Index>> nbr_buf;
    nbr_buf.reserve(128);

    Rng local_rng(42);
    Rng &active_rng = rng ? *rng : local_rng;

    for (Index step = 0; step + 1 < n; ++step) {
        const Index v = q.pop_min();
        factor.order.push_back(v);

        Float total_weight = collect_neighbors(v, G, done, q, star_buf, nbr_buf);
        done[v] = 1;

        if (total_weight <= static_cast<Float>(0)) {
            continue;
        }

        detail::write_column(v, step, total_weight, nbr_buf, factor);

        if (exact_mode) {
            add_exact_clique(G, q, nbr_buf, total_weight);
        } else {
            sample_clique(G, q, nbr_buf, total_weight, samples, active_rng);
        }
    }

    // Nullspace vertex
    const Index last_v = q.pop_min();
    factor.order.push_back(last_v);

    detail::permute_rows(factor);

    return factor;
}

template <typename Float = double, typename Index = std::size_t>
inline CholeskyFactor<Float, Index> ac1(const Graph<Float, Index> &G, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    return factorize<Float, Index, std::mt19937_64>(G, 1, false, &rng);
}

template <typename Float = double, typename Index = std::size_t, typename Rng = std::mt19937_64>
inline CholeskyFactor<Float, Index> ac1(const Graph<Float, Index> &G, Rng &rng) {
    return factorize<Float, Index, Rng>(G, 1, false, &rng);
}

template <typename Float = double, typename Index = std::size_t>
inline CholeskyFactor<Float, Index> ac2(const Graph<Float, Index> &G, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    return factorize<Float, Index, std::mt19937_64>(G, 2, false, &rng);
}

template <typename Float = double, typename Index = std::size_t, typename Rng = std::mt19937_64>
inline CholeskyFactor<Float, Index> ac2(const Graph<Float, Index> &G, Rng &rng) {
    return factorize<Float, Index, Rng>(G, 2, false, &rng);
}

template <typename Float = double, typename Index = std::size_t>
inline CholeskyFactor<Float, Index> exact(const Graph<Float, Index> &G) {
    return factorize<Float, Index, std::mt19937_64>(G, 1, true, nullptr);
}

} // namespace approxchol
