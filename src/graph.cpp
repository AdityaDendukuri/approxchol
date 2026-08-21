#include "approxchol/graph.h"
#include "approxchol/degree_queue.h"

#include <algorithm>

namespace approxchol {

void add_edge(Graph &G, idx u, idx v, double weight) {
  G[u].push_back({v, weight, 1});
  G[v].push_back({u, weight, 1});
}

double collect_neighbors(idx v, Graph &G,
                         const std::vector<std::uint8_t> &done, DegreeQueue &q,
                         std::vector<Edge> &star_buf,
                         std::vector<Neighbor> &nbr_out) {
  // 1. Filter out dead edges to already-eliminated nodes
  star_buf.clear();
  for (const auto &e : G[v]) {
    if (done[e.to]) {
      continue;
    }
    star_buf.push_back(e);
    q.rekey(e.to, q.degree[e.to] - 1);
  }

  G[v].clear();
  G[v].shrink_to_fit();

  if (star_buf.empty()) {
    return 0.0;
  }

  // 2. Sort star by neighbor ID to group duplicate parallel fill-ins
  std::sort(star_buf.begin(), star_buf.end(),
            [](const Edge &a, const Edge &b) { return a.to < b.to; });

  // 3. Linear pass to merge weights and multiplicity counts into unique neighbors
  nbr_out.clear();
  double total_weight = 0.0;

  for (const auto &e : star_buf) {
    if (!nbr_out.empty() && nbr_out.back().to == e.to) {
      nbr_out.back().weight += e.weight;
      nbr_out.back().count = static_cast<std::uint8_t>(
          std::min<idx>(255, nbr_out.back().count + e.count));
    } else {
      nbr_out.push_back({e.to, e.weight, e.count});
    }
    total_weight += e.weight;
  }

  return total_weight;
}

void add_exact_clique(Graph &G, DegreeQueue &q,
                      const std::vector<Neighbor> &nbr, double total_weight) {
  for (idx a = 0; a < nbr.size(); ++a) {
    for (idx b = a + 1; b < nbr.size(); ++b) {
      double w_exact = (nbr[a].weight * nbr[b].weight) / total_weight;
      idx u = nbr[a].to;
      idx j = nbr[b].to;

      G[u].push_back({j, w_exact, 1});
      G[j].push_back({u, w_exact, 1});
      q.rekey(u, q.degree[u] + 1);
      q.rekey(j, q.degree[j] + 1);
    }
  }
}

void sample_clique(Graph &G, DegreeQueue &q,
                   std::vector<Neighbor> &nbr, double total_weight,
                   idx sample_limit, std::mt19937_64 &rng) {
  // Sort neighbors light-to-heavy by weight
  std::sort(nbr.begin(), nbr.end(), [](const Neighbor &a, const Neighbor &b) {
    return a.weight < b.weight;
  });

  double rest = total_weight;
  std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

  for (idx a = 0; a + 1 < nbr.size(); ++a) {
    const auto &nbr_a = nbr[a];
    rest -= nbr_a.weight;

    if (rest <= 0.0) {
      break;
    }

    const idx draws = std::min<idx>(nbr_a.count, sample_limit);
    const double w_bar = nbr_a.weight / static_cast<double>(draws);
    const double w_new = w_bar * (rest / total_weight);

    for (idx s = 0; s < draws; ++s) {
      double target = unit_dist(rng) * rest;

      idx b = a + 1;
      while (b + 1 < nbr.size() && target >= nbr[b].weight) {
        target -= nbr[b].weight;
        ++b;
      }

      idx u = nbr_a.to;
      idx j = nbr[b].to;

      // Sampled fill-in edge is added with count = 1
      G[u].push_back({j, w_new, 1});
      G[j].push_back({u, w_new, 1});
      q.rekey(u, q.degree[u] + 1);
      q.rekey(j, q.degree[j] + 1);
    }
  }
}

} // namespace approxchol
