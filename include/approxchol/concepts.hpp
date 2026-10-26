/// @file approxchol/concepts.hpp  Bring your own graph type.
#pragma once
#include "approxchol/factorize.hpp"
#include <ranges>
#include <utility>
namespace approxchol {
/// A random-access range of neighbor lists whose edges have `.to` and `.weight`.
template <class G>
concept weighted_graph =
    std::ranges::random_access_range<G> && std::ranges::sized_range<G> &&
    requires(const std::ranges::range_value_t<std::ranges::range_reference_t<G>> &e) {
        { e.to } -> std::convertible_to<std::size_t>;
        { e.weight } -> std::convertible_to<double>;
    };

template <weighted_graph G>
adjacency_list<double> to_adjacency_list(const G &g) {
    adjacency_list<double> out(std::ranges::size(g));
    for (std::size_t u = 0; u < out.size(); ++u)
        for (const auto &e : g[u]) out[u].push_back({std::size_t(e.to), double(e.weight)});
    return out;
}
// For user graph types only; an adjacency_list goes straight to factorize.hpp.
template <class G> constexpr bool foreign = weighted_graph<G>;
template <class F, std::integral I> constexpr bool foreign<adjacency_list<F, I>> = false;
template <class G, class... A> requires foreign<G> auto ac1(const G &g, A &&...a) { return ac1(to_adjacency_list(g), std::forward<A>(a)...); }
template <class G, class... A> requires foreign<G> auto ac2(const G &g, A &&...a) { return ac2(to_adjacency_list(g), std::forward<A>(a)...); }
template <class G> requires foreign<G> auto exact(const G &g) { return exact(to_adjacency_list(g)); }
} // namespace approxchol
