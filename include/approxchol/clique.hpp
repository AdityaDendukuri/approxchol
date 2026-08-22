#pragma once

#include "approxchol/degree_queue.hpp"
#include "approxchol/types.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace approxchol
{

template <typename Float = double, typename Index = std::size_t,
          typename Queue = DegreeQueue<Index, Float>>
inline Float collect_neighbors(Index v, Graph<Float, Index> &G,
                               const std::vector<std::uint8_t> &done, Queue &q,
                               std::vector<Edge<Float, Index>> &star_buf,
                               std::vector<Neighbor<Float, Index>> &nbr_out)
{
    star_buf.clear();
    for (const auto &e : G[v]) {
        if (done[e.to]) {
            continue;
        }
        star_buf.push_back(e);
        q.rekey(e.to, q.degree[e.to] - e.count);
    }

    G[v].clear();
    G[v].shrink_to_fit();

    if (star_buf.empty()) {
        return static_cast<Float>(0);
    }

    std::sort(star_buf.begin(), star_buf.end(),
              [](const Edge<Float, Index> &a, const Edge<Float, Index> &b) { return a.to < b.to; });

    nbr_out.clear();
    Float total_weight = static_cast<Float>(0);

    for (const auto &e : star_buf) {
        if (!nbr_out.empty() && nbr_out.back().to == e.to) {
            nbr_out.back().weight += e.weight;
            nbr_out.back().count =
                static_cast<std::uint8_t>(std::min<Index>(255, nbr_out.back().count + e.count));
        } else {
            nbr_out.push_back({e.to, e.weight, e.count});
        }
        total_weight += e.weight;
    }

    return total_weight;
}

template <typename Float = double, typename Index = std::size_t,
          typename Queue = DegreeQueue<Index, Float>>
inline void add_exact_clique(Graph<Float, Index> &G, Queue &q,
                             const std::vector<Neighbor<Float, Index>> &nbr, Float total_weight)
{
    for (Index a = 0; a < nbr.size(); ++a) {
        for (Index b = a + 1; b < nbr.size(); ++b) {
            Float w_exact = (nbr[a].weight * nbr[b].weight) / total_weight;
            Index u = nbr[a].to;
            Index j = nbr[b].to;

            G[u].push_back({j, w_exact, 1});
            G[j].push_back({u, w_exact, 1});
            q.rekey(u, q.degree[u] + 1);
            q.rekey(j, q.degree[j] + 1);
        }
    }
}

template <typename Float = double, typename Index = std::size_t,
          typename Queue = DegreeQueue<Index, Float>, typename Rng = std::mt19937_64>
inline void sample_clique(Graph<Float, Index> &G, Queue &q,
                          std::vector<Neighbor<Float, Index>> &nbr, Float total_weight,
                          Index sample_limit, Rng &rng)
{
    std::sort(nbr.begin(), nbr.end(),
              [](const Neighbor<Float, Index> &a, const Neighbor<Float, Index> &b) {
                  return a.weight < b.weight;
              });

    Float rest = total_weight;
    std::uniform_real_distribution<Float> unit_dist(static_cast<Float>(0), static_cast<Float>(1));

    for (Index a = 0; a + 1 < nbr.size(); ++a) {
        const auto &nbr_a = nbr[a];
        rest -= nbr_a.weight;

        if (rest <= static_cast<Float>(0)) {
            break;
        }

        const Index draws = std::min<Index>(nbr_a.count, sample_limit);
        const Float w_bar = nbr_a.weight / static_cast<Float>(draws);
        const Float w_new = w_bar * (rest / total_weight);

        for (Index s = 0; s < draws; ++s) {
            Float target = unit_dist(rng) * rest;

            Index b = a + 1;
            while (b + 1 < nbr.size() && target >= nbr[b].weight) {
                target -= nbr[b].weight;
                ++b;
            }

            Index u = nbr_a.to;
            Index j = nbr[b].to;

            G[u].push_back({j, w_new, 1});
            G[j].push_back({u, w_new, 1});
            q.rekey(u, q.degree[u] + 1);
            q.rekey(j, q.degree[j] + 1);
        }
    }
}

} // namespace approxchol
