/// @file approxchol/clique.hpp
/// @brief Exact and randomized clique reduction (star-mesh transformation) on multigraphs.
#pragma once

#include "approxchol/degree_queue.hpp"
#include "approxchol/types.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

namespace approxchol {

/// @brief Gather the neighbors of `v`, merging parallel edges, and return their total weight.
template <typename Weight = double, std::integral Index = std::size_t,
          typename Queue = basic_degree_queue<Index>>
inline Weight collect_neighbors(Index v, std::vector<std::vector<multi_edge<Weight, Index>>> &G,
                                const std::vector<std::uint8_t> &done, Queue &q,
                                std::vector<multi_edge<Weight, Index>> &star_buf,
                                std::vector<multi_edge<Weight, Index>> &nbr_out) {
    star_buf.clear();
    for (const auto &e : G[v]) {
        if (done[e.to]) {
            continue;
        }
        star_buf.push_back(e);
        q.rekey(e.to, q.degree_of(e.to) - e.count);
    }

    G[v].clear();
    G[v].shrink_to_fit();

    if (star_buf.empty()) {
        return static_cast<Weight>(0);
    }

    std::sort(star_buf.begin(), star_buf.end(),
              [](const multi_edge<Weight, Index> &a, const multi_edge<Weight, Index> &b) {
                  return a.to < b.to;
              });

    nbr_out.clear();
    Weight total_weight = static_cast<Weight>(0);

    for (const auto &e : star_buf) {
        if (!nbr_out.empty() && nbr_out.back().to == e.to) {
            nbr_out.back().weight += e.weight;
            nbr_out.back().count =
                static_cast<std::uint8_t>(std::min<Index>(255, nbr_out.back().count + e.count));
        } else {
            nbr_out.push_back(e);
        }
        total_weight += e.weight;
    }

    return total_weight;
}

/// @brief Add the exact clique that eliminating a vertex creates among its neighbors.
template <typename Weight = double, std::integral Index = std::size_t,
          typename Queue = basic_degree_queue<Index>>
inline void add_exact_clique(std::vector<std::vector<multi_edge<Weight, Index>>> &G, Queue &q,
                             const std::vector<multi_edge<Weight, Index>> &nbr,
                             Weight total_weight) {
    for (Index a = 0; a < nbr.size(); ++a) {
        for (Index b = a + 1; b < nbr.size(); ++b) {
            Weight w_exact = (nbr[a].weight * nbr[b].weight) / total_weight;
            Index u = nbr[a].to;
            Index j = nbr[b].to;

            G[u].push_back({j, w_exact, 1});
            G[j].push_back({u, w_exact, 1});
            q.rekey(u, q.degree_of(u) + 1);
            q.rekey(j, q.degree_of(j) + 1);
        }
    }
}

/// @brief Add a sampled sparse approximation of that clique, with at most `sample_limit` draws
/// per neighbor.
template <typename Weight = double, std::integral Index = std::size_t,
          typename Queue = basic_degree_queue<Index>, typename Rng = rng64>
inline void sample_clique(std::vector<std::vector<multi_edge<Weight, Index>>> &G, Queue &q,
                          std::vector<multi_edge<Weight, Index>> &nbr, Weight total_weight,
                          std::type_identity_t<Index> sample_limit, Rng &rng) {
    std::sort(nbr.begin(), nbr.end(),
              [](const multi_edge<Weight, Index> &a, const multi_edge<Weight, Index> &b) {
                  return a.weight < b.weight;
              });

    Weight rest = total_weight;
    std::uniform_real_distribution<Weight> unit_dist(static_cast<Weight>(0),
                                                     static_cast<Weight>(1));

    for (Index a = 0; a + 1 < nbr.size(); ++a) {
        const auto &nbr_a = nbr[a];
        rest -= nbr_a.weight;

        if (rest <= static_cast<Weight>(0)) {
            break;
        }

        const Index draws = std::min<Index>(nbr_a.count, sample_limit);
        const Weight w_bar = nbr_a.weight / static_cast<Weight>(draws);
        const Weight w_new = w_bar * (rest / total_weight);

        for (Index s = 0; s < draws; ++s) {
            Weight target = unit_dist(rng) * rest;

            Index b = a + 1;
            while (b + 1 < nbr.size() && target >= nbr[b].weight) {
                target -= nbr[b].weight;
                ++b;
            }

            Index u = nbr_a.to;
            Index j = nbr[b].to;

            G[u].push_back({j, w_new, 1});
            G[j].push_back({u, w_new, 1});
            q.rekey(u, q.degree_of(u) + 1);
            q.rekey(j, q.degree_of(j) + 1);
        }
    }
}

/// @brief Replace the star-mesh clique by a random spanning tree of it, reweighted
/// so the elimination stays unbiased in expectation.
///
/// Eliminating `v` adds a clique on its neighbours with weights \f$w_{ij} = c_i c_j / C\f$,
/// where \f$c_i\f$ is the conductance from `v` to `i` and \f$C = \sum_i c_i\f$. Edge
/// \f$ij\f$ enters the tree with probability \f$p_{ij} = (c_i + c_j)/C\f$ and is reweighted
/// to the harmonic mean \f$c_i c_j / (c_i + c_j)\f$, so \f$\mathbb{E}[\widetilde L^{(v)}] =
/// \mathrm{Sc}(L)\f$. The walk is capped: past the cap the remaining vertices attach to the
/// heaviest neighbour, which keeps a spanning tree but loses exactness for that elimination.
///
/// @param G Adjacency being eliminated; sampled edges are appended.
/// @param q Degree queue, rekeyed for each endpoint touched.
/// @param nbr Neighbours of the eliminated vertex, with their conductances.
/// @param total_weight \f$C\f$, the sum of those conductances.
/// @param rng Random source.
template <typename Weight, std::integral Index, typename Queue, typename Rng>
inline void sample_clique_tree(std::vector<std::vector<multi_edge<Weight, Index>>> &G, Queue &q,
                               const std::vector<multi_edge<Weight, Index>> &nbr,
                               Weight total_weight, Rng &rng, std::size_t trees = 1) {
    const std::size_t degree = nbr.size();
    if (degree < 2 || !(total_weight > Weight(0)) || trees == 0) {
        return;
    }

    std::vector<Weight> cumulative(degree);
    Weight running = Weight(0);
    std::size_t heaviest = 0;
    for (std::size_t i = 0; i < degree; ++i) {
        running += nbr[i].weight;
        cumulative[i] = running;
        if (nbr[i].weight > nbr[heaviest].weight) {
            heaviest = i;
        }
    }
    if (!(running > Weight(0))) {
        return;
    }

    std::uniform_real_distribution<Weight> unit(Weight(0), Weight(1));
    const auto draw = [&]() -> std::size_t {
        const Weight target = unit(rng) * running;
        std::size_t low = 0;
        std::size_t high = degree - 1;
        while (low < high) {
            const std::size_t mid = low + ((high - low) / 2);
            if (cumulative[mid] < target) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    };

    const auto connect = [&](std::size_t a, std::size_t b) {
        const Weight wa = nbr[a].weight;
        const Weight wb = nbr[b].weight;
        const Weight sum = wa + wb;
        if (!(sum > Weight(0))) {
            return;
        }
        // Harmonic mean w_ij/p_ij, split across the trees so the average of
        // `trees` independent samples remains unbiased.
        const Weight weight = (wa * wb) / (sum * static_cast<Weight>(trees));
        const Index u = nbr[a].to;
        const Index v = nbr[b].to;
        G[u].push_back({v, weight, 1});
        G[v].push_back({u, weight, 1});
        q.rekey(u, q.degree_of(u) + 1);
        q.rekey(v, q.degree_of(v) + 1);
    };

    for (std::size_t tree = 0; tree < trees; ++tree) {
        std::vector<char> visited(degree, 0);
        std::size_t current = draw();
        visited[current] = 1;
        std::size_t remaining = degree - 1;

        // Coupon-collector expectation is d*H_d for uniform weights; the cap leaves
        // generous room for moderate skew before the deterministic fallback.
        const std::size_t budget = (8 * degree * (1 + degree / 4)) + 64;
        for (std::size_t step = 0; step < budget && remaining > 0; ++step) {
            const std::size_t next = draw();
            if (next == current) {
                continue;
            }
            if (visited[next] == 0) {
                connect(current, next);
                visited[next] = 1;
                --remaining;
            }
            current = next;
        }
        if (remaining > 0) {
            for (std::size_t i = 0; i < degree; ++i) {
                if (visited[i] == 0 && i != heaviest) {
                    connect(heaviest, i);
                    visited[i] = 1;
                }
            }
        }
    }
}

} // namespace approxchol
