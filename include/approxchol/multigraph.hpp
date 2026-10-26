/// @file approxchol/multigraph.hpp
/// @brief Weighted multigraph with edge multiplicities, used by approximate Cholesky.
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace approxchol {

/// Weighted multi-edge with multiplicity count.
template <typename Weight = double, std::integral Index = std::size_t>
struct multi_edge {
    Index to{};
    Weight weight{1};
    std::uint8_t count = 1;
};

/// @brief Weighted multigraph representation supporting parallel edges and integer counts.
template <typename Weight = double, std::integral Index = std::size_t>
class multigraph {
  public:
    using weight_type = Weight;
    using index_type = Index;
    using edge_type = multi_edge<Weight, Index>;

    /// Construct an empty multigraph with n vertices.
    explicit multigraph(Index n = 0) : adj_(n) {}

    /// Number of vertices.
    [[nodiscard]] Index n_vertices() const noexcept { return static_cast<Index>(adj_.size()); }

    /// Add an undirected multi-edge between u and v.
    void add_edge(Index u, Index v, Weight weight = Weight{1}, std::uint8_t count = 1) {
        if (u >= adj_.size() || v >= adj_.size()) {
            throw std::out_of_range("multigraph::add_edge: vertex index out of range");
        }
        adj_[u].push_back({v, weight, count});
        adj_[v].push_back({u, weight, count});
    }

    /// Const access to neighbor list of vertex u.
    [[nodiscard]] const std::vector<edge_type> &neighbors(Index u) const {
        if (u >= adj_.size()) {
            throw std::out_of_range("multigraph::neighbors: vertex index out of range");
        }
        return adj_[u];
    }

    /// Raw underlying adjacency list.
    [[nodiscard]] const std::vector<std::vector<edge_type>> &adjacency() const noexcept {
        return adj_;
    }

    /// Mutable raw underlying adjacency list.
    [[nodiscard]] std::vector<std::vector<edge_type>> &adjacency() noexcept { return adj_; }

    /// Subscript operator for vertex neighbors.
    [[nodiscard]] const std::vector<edge_type> &operator[](Index u) const { return adj_[u]; }
    [[nodiscard]] std::vector<edge_type> &operator[](Index u) { return adj_[u]; }

  private:
    std::vector<std::vector<edge_type>> adj_;
};

} // namespace approxchol
