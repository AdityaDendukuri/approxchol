#include "approxchol/solve.h"
#include "random_graph.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <N> [repeats=5]\n";
    return 1;
  }

  approxchol::idx n = std::stoull(argv[1]);
  int repeats = (argc >= 3) ? std::atoi(argv[2]) : 5;

  approxchol::Graph G = approxchol::make_random_laplacian(n, 6, 42);

  // Warmup
  auto f1_w = approxchol::ac1(G, 42);
  auto f2_w = approxchol::ac2(G, 42);

  std::vector<double> ac1_ms(repeats);
  for (int r = 0; r < repeats; ++r) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto f = approxchol::ac1(G, 42 + r);
    auto t1 = std::chrono::high_resolution_clock::now();
    ac1_ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  std::vector<double> ac2_ms(repeats);
  for (int r = 0; r < repeats; ++r) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto f = approxchol::ac2(G, 42 + r);
    auto t1 = std::chrono::high_resolution_clock::now();
    ac2_ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  auto avg = [](const std::vector<double> &v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  };

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "cpp\t" << n << "\t" << avg(ac1_ms) << "\t" << avg(ac2_ms) << "\n";

  return 0;
}
