/// @file approxchol/concepts.hpp
/// @brief Bring-your-own-graph interface: any weighted adjacency structure can be factorized.
#pragma once

#include "approxchol/factorize.hpp"
#include "approxchol/types.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <type_traits>

namespace approxchol {

/// @brief An edge with an integer endpoint `to` and a numeric `weight`.
template <class E>
concept weighted_edge = requires(const E &e) {
    { e.to } -> std::convertible_to<std::size_t>;
    { e.weight } -> std::convertible_to<double>;
};

/// @brief A random-access range of neighbor lists, one per vertex, of weighted edges.
/// Each undirected edge appears in both endpoints' lists, as for `add_edge`.
template <class G>
concept weighted_graph =
    std::ranges::random_access_range<G> && std::ranges::sized_range<G> &&
    std::ranges::input_range<std::ranges::range_reference_t<G>> &&
    weighted_edge<std::ranges::range_value_t<std::ranges::range_reference_t<G>>>;

namespace detail {

template <class G>
inline constexpr bool is_adjacency_list = false;

template <typename Float, std::integral Index>
inline constexpr bool is_adjacency_list<adjacency_list<Float, Index>> = true;

/// A user graph, not one of ours: the overloads below convert it first.
template <class G>
concept foreign_graph = weighted_graph<G> && !is_adjacency_list<std::remove_cvref_t<G>>;

} // namespace detail

/// @brief Copy any `weighted_graph` into the library's adjacency list.
template <typename Float = double, std::integral Index = std::size_t, weighted_graph G>
[[nodiscard]] inline adjacency_list<Float, Index> to_adjacency_list(const G &g) {
    adjacency_list<Float, Index> out(std::ranges::size(g));
    for (std::size_t u = 0; u < out.size(); ++u) {
        for (const auto &e : g[u]) {
            out[u].push_back({static_cast<Index>(e.to), static_cast<Float>(e.weight), 1});
        }
    }
    return out;
}

template <detail::foreign_graph G, typename Rng>
inline cholesky_factor<double, std::size_t> ac1(const G &g, Rng &rng) {
    return ac1(to_adjacency_list(g), rng);
}

template <detail::foreign_graph G, typename Rng>
inline cholesky_factor<double, std::size_t> ac2(const G &g, Rng &rng) {
    return ac2(to_adjacency_list(g), rng);
}

template <detail::foreign_graph G, typename Rng>
inline cholesky_factor<double, std::size_t> act(const G &g, Rng &rng, std::size_t trees = 1) {
    return act(to_adjacency_list(g), rng, trees);
}

template <detail::foreign_graph G>
inline cholesky_factor<double, std::size_t> exact(const G &g) {
    return exact(to_adjacency_list(g));
}

static_assert(weighted_graph<adjacency_list<double, std::size_t>>);

} // namespace approxchol
