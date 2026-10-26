/// @file approxchol/degree_queue.hpp
/// @brief Bucket queue keyed by degree, for minimum-degree elimination.
#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace approxchol {

/// @brief Bucket-based degree priority queue maintaining vertices ordered by integer degree.
/// Provides O(1) insert, erase, rekey, and O(1) amortized pop_min for bounded integer degrees.
template <std::integral Index = std::size_t>
class basic_degree_queue {
  public:
    using index_type = Index;
    static constexpr Index none = static_cast<Index>(-1);

    /// Construct an empty queue with capacity for n vertices.
    explicit basic_degree_queue(Index n = 0) {
        head_.assign(static_cast<std::size_t>(n) + 1, none);
        next_.assign(static_cast<std::size_t>(n), none);
        prev_.assign(static_cast<std::size_t>(n), none);
        degree_.assign(static_cast<std::size_t>(n), 0);
    }

    /// Insert vertex v with degree d.
    void insert(Index v, Index d) {
        if (d >= head_.size()) {
            head_.resize(std::max(static_cast<std::size_t>(d) + 1, 2 * head_.size()), none);
        }
        next_[v] = head_[d];
        prev_[v] = none;
        if (head_[d] != none) {
            prev_[head_[d]] = v;
        }
        head_[d] = v;
        degree_[v] = d;
        if (d < low_) {
            low_ = d;
        }
        ++size_;
    }

    /// Erase vertex v from the queue.
    void erase(Index v) {
        const Index d = degree_[v];
        if (prev_[v] == none) {
            head_[d] = next_[v];
        } else {
            next_[prev_[v]] = next_[v];
        }
        if (next_[v] != none) {
            prev_[next_[v]] = prev_[v];
        }
        next_[v] = none;
        prev_[v] = none;
        --size_;
    }

    /// Update the degree of vertex v.
    void rekey(Index v, Index d) {
        erase(v);
        insert(v, d);
    }

    /// Pop vertex with minimum degree.
    Index pop_min() {
        if (empty()) {
            throw std::runtime_error("degree_queue::pop_min: queue is empty");
        }
        while (low_ < head_.size() && head_[low_] == none) {
            ++low_;
        }
        if (low_ == head_.size()) {
            throw std::runtime_error("degree_queue::pop_min: corrupted bucket state");
        }
        const Index v = head_[low_];
        erase(v);
        return v;
    }

    /// Degree of vertex v.
    [[nodiscard]] Index degree_of(Index v) const { return degree_[v]; }

    /// Check if queue is empty.
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// Number of active elements.
    [[nodiscard]] Index size() const noexcept { return size_; }

    /// Total capacity.
    [[nodiscard]] Index capacity() const noexcept { return static_cast<Index>(degree_.size()); }

  private:
    std::vector<Index> head_;
    std::vector<Index> next_;
    std::vector<Index> prev_;
    std::vector<Index> degree_;
    Index low_ = 0;
    Index size_ = 0;
};

using degree_queue = basic_degree_queue<std::size_t>;
using degree_queue_32 = basic_degree_queue<std::uint32_t>;

} // namespace approxchol
