#include "approxchol/multigraph.h"
#include "approxchol/degree_queue.h"

#include <algorithm>

namespace approxchol {

void add_edge(MultiGraph &G, idx u, idx v, double weight, idx count) {
  double split_weight = weight / static_cast<double>(count);
  for (idx c = 0; c < count; ++c) {
    G[u].push_back({v, split_weight});
    G[v].push_back({u, split_weight});
  }
}

double collect_neighbors(idx v, MultiGraph &G,
                         const std::vector<std::uint8_t> &done, DegreeQueue &q,
                         std::vector<Edge> &star_buf,
                         std::vector<Neighbor> &nbr_out) {

  // filter out dead edges to already-eliminated nodes
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

  // sort star by neighbor id to group parallel multi-edges together
  std::sort(star_buf.begin(), star_buf.end(),
            [](const Edge &a, const Edge &b) { return a.to < b.to; });

  // merge parallel copies into distinct neighbors
  nbr_out.clear();
  double total_weight = 0.0;

  for (const auto &e : star_buf) {
    if (!nbr_out.empty() && nbr_out.back().to == e.to) {
      nbr_out.back().weight += e.weight;
      ++nbr_out.back().count;
    } else {
      nbr_out.push_back({e.to, e.weight, 1});
    }
    total_weight += e.weight;
  }

  return total_weight;
}

void add_exact_clique(MultiGraph &G, DegreeQueue &q,
                      const std::vector<Neighbor> &nbr, double total_weight) {
  for (idx a = 0; a < nbr.size(); ++a) {
    for (idx b = a + 1; b < nbr.size(); ++b) {
      double w_exact = (nbr[a].weight * nbr[b].weight) / total_weight;
      idx u = nbr[a].to;
      idx j = nbr[b].to;

      G[u].push_back({j, w_exact});
      G[j].push_back({u, w_exact});
      q.rekey(u, q.degree[u] + 1);
      q.rekey(j, q.degree[j] + 1);
    }
  }
}

void add_sampled_clique(MultiGraph &G, DegreeQueue &q,
                        std::vector<Neighbor> &nbr, double total_weight,
                        idx sample_limit, std::mt19937_64 &rng) {
  // Sort neighbors light-to-heavy by weight
  std::sort(nbr.begin(), nbr.end(), [](const Neighbor &a, const Neighbor &b) {
    return a.weight < b.weight;
  });

  double rest = total_weight;

  for (idx a = 0; a + 1 < nbr.size(); ++a) {
    const auto &nbr_a = nbr[a];
    rest -= nbr_a.weight;

    if (rest <= 0.0) {
      break;
    }

    const idx t = std::min(nbr_a.count, sample_limit);
    const double w_bar = nbr_a.weight / static_cast<double>(t);
    const double w_new = w_bar * (rest / total_weight);

    for (idx h = 0; h < t; ++h) {
      std::uniform_real_distribution<double> dist(0.0, rest);
      double target = dist(rng);

      idx b = a + 1;
      while (b + 1 < nbr.size() && target >= nbr[b].weight) {
        target -= nbr[b].weight;
        ++b;
      }

      idx u = nbr_a.to;
      idx j = nbr[b].to;

      G[u].push_back({j, w_new});
      G[j].push_back({u, w_new});
      q.rekey(u, q.degree[u] + 1);
      q.rekey(j, q.degree[j] + 1);
    }
  }
}

} // namespace approxchol
