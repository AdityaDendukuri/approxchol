#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace approxchol {

using idx = std::size_t;

struct DegreeQueue;

struct Edge {
  idx to;
  double weight;
};

using MultiGraph = std::vector<std::vector<Edge>>;

struct Neighbor {
  idx to;
  double weight;
  idx count;
};

void add_edge(MultiGraph &G, idx u, idx v, double weight, idx count = 1);

double collect_neighbors(idx v, MultiGraph &G,
                         const std::vector<std::uint8_t> &done, DegreeQueue &q,
                         std::vector<Edge> &star_buf,
                         std::vector<Neighbor> &nbr_out);

void add_exact_clique(MultiGraph &G, DegreeQueue &q,
                      const std::vector<Neighbor> &nbr, double total_weight);

void add_sampled_clique(MultiGraph &G, DegreeQueue &q,
                        std::vector<Neighbor> &nbr, double total_weight,
                        idx sample_limit, std::mt19937_64 &rng);

} // namespace approxchol
