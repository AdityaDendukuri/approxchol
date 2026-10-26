/// @file approxchol/factorize.hpp  AC/AC2 of Gao, Kyng, Spielman (2023), arXiv:2303.00709.
#pragma once
#include "approxchol/degree_queue.hpp"
#include "approxchol/types.hpp"
#include <algorithm>
#include <cmath>
namespace approxchol {
/// Eliminate in minimum-degree order. Each eliminated vertex's clique is replaced by
/// `samples` draws per neighbor (1 = AC, 2 = AC2), or kept whole when `exact_mode`.
template <typename Float, std::integral Index, typename Rng>
cholesky_factor<Float, Index> factorize(adjacency_list<Float, Index> G, Index samples,
                                        bool exact_mode, Rng &rng) {
    const Index n = static_cast<Index>(G.size());
    degree_queue<Index> q(n);
    for (Index i = 0; i < n; ++i) {
        Index d = 0;
        for (auto &e : G[i]) d += (e.count = exact_mode ? 1 : static_cast<std::uint8_t>(samples));
        q.insert(i, d);
    }
    auto link = [&](Index u, Index v, Float w) {
        G[u].push_back({v, w, 1}), G[v].push_back({u, w, 1});
        q.rekey(u, q.degree[u] + 1), q.rekey(v, q.degree[v] + 1);
    };
    std::vector<std::uint8_t> done(n, 0);
    std::vector<edge<Float, Index>> star, nbr;
    std::uniform_real_distribution<Float> unit(0, 1);
    cholesky_factor<Float, Index> F; F.order.reserve(n), star.reserve(128), nbr.reserve(128);
    for (Index step = 0; step < n; ++step) {
        const Index v = q.pop_min();
        F.order.push_back(v);
        star.clear(), nbr.clear();
        for (const auto &e : G[v])
            if (!done[e.to]) star.push_back(e), q.rekey(e.to, q.degree[e.to] - e.count);
        G[v].clear(), G[v].shrink_to_fit(), done[v] = 1;
        std::sort(star.begin(), star.end(), [](auto &x, auto &y) { return x.to < y.to; }); // merge parallel edges
        Float total = 0;
        for (const auto &e : star) {
            total += e.weight;
            if (nbr.empty() || nbr.back().to != e.to) nbr.push_back(e);
            else nbr.back().weight += e.weight, nbr.back().count = std::min(255, nbr.back().count + e.count);
        }
        if (step + 1 == n || total <= 0) { F.start.push_back(F.row.size()); continue; } // nullspace
        const Float s = std::sqrt(total); F.row.push_back(v), F.value.push_back(s);
        for (const auto &e : nbr) F.row.push_back(e.to), F.value.push_back(-e.weight / s);
        F.start.push_back(F.row.size());
        if (exact_mode) {
            for (std::size_t a = 0; a < nbr.size(); ++a)
                for (std::size_t b = a + 1; b < nbr.size(); ++b)
                    link(nbr[a].to, nbr[b].to, nbr[a].weight * nbr[b].weight / total);
            continue;
        }
        // Each lighter neighbor draws partners among the heavier ones in proportion to
        // weight, reweighted so the expected Schur complement is exact.
        std::sort(nbr.begin(), nbr.end(), [](auto &x, auto &y) { return x.weight < y.weight; });
        Float rest = total;
        for (std::size_t a = 0; a + 1 < nbr.size() && (rest -= nbr[a].weight) > 0; ++a) {
            const Index draws = std::min<Index>(nbr[a].count, samples);
            const Float w = nbr[a].weight / draws * (rest / total);
            for (Index d = 0; d < draws; ++d) {
                Float target = unit(rng) * rest;
                std::size_t b = a + 1;
                while (b + 1 < nbr.size() && target >= nbr[b].weight) target -= nbr[b++].weight;
                link(nbr[a].to, nbr[b].to, w);
            }
        }
    }
    std::vector<Index> position(n);
    for (Index k = 0; k < n; ++k) position[F.order[k]] = k;
    for (auto &r : F.row) r = position[r];
    return F;
}
template <class F, std::integral I, class R> auto ac1(const adjacency_list<F, I> &G, R &r) { return factorize(G, I{1}, false, r); }
template <class F, std::integral I, class R> auto ac2(const adjacency_list<F, I> &G, R &r) { return factorize(G, I{2}, false, r); }
template <class F, std::integral I> auto ac1(const adjacency_list<F, I> &G, std::uint64_t seed = 42) { rng64 r(seed); return ac1(G, r); }
template <class F, std::integral I> auto ac2(const adjacency_list<F, I> &G, std::uint64_t seed = 42) { rng64 r(seed); return ac2(G, r); }
template <class F, std::integral I> auto exact(const adjacency_list<F, I> &G) { rng64 r; return factorize(G, I{1}, true, r); }
} // namespace approxchol
