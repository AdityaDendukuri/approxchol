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
  std::uint8_t count = 1;
};

using Graph = std::vector<std::vector<Edge>>;

struct Neighbor {
  idx to;
  double weight;
  std::uint8_t count = 1;
};

void add_edge(Graph &G, idx u, idx v, double weight);

double collect_neighbors(idx v, Graph &G,
                         const std::vector<std::uint8_t> &done,
                         DegreeQueue &q, std::vector<Edge> &star_buf,
                         std::vector<Neighbor> &nbr_out);

void add_exact_clique(Graph &G, DegreeQueue &q,
                      const std::vector<Neighbor> &nbr, double total_weight);

void sample_clique(Graph &G, DegreeQueue &q,
                   std::vector<Neighbor> &nbr, double total_weight,
                   idx sample_limit, std::mt19937_64 &rng);

} // namespace approxchol
