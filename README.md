# approxchol

Header-only C++20 randomized approximate Cholesky for graph Laplacians and SDD
M-matrices. Use the factor as a preconditioner (e.g. for CG). No dependencies.

Implements Gao, Kyng & Spielman, *Robust and Practical Solution of Laplacian
Equations by Approximate Elimination* (2023), [arXiv:2303.00709](https://arxiv.org/abs/2303.00709).

## Build

```bash
cmake -S . -B build && cmake --build build -j
./build/test_approxchol
```

Or just add `include/` to your include path and `#include "approxchol/approxchol.hpp"`.

## Use

```cpp
#include "approxchol/approxchol.hpp"

approxchol::adjacency_list<double> G(n);            // weighted graph, one list per vertex
approxchol::add_edge(G, u, v, w);                   // undirected edge u--v, weight w

approxchol::rng64 rng(42);
auto F = approxchol::ac2(G, rng);                   // approximate factor, P L P^T ~ L_G
std::vector<double> x = approxchol::solve(F, b);    // apply (L L^T)^{-1} b
```

## Factorizations

| Call | What |
| --- | --- |
| `ac1(G, rng)` | GKS AC: one clique sample per edge |
| `ac2(G, rng)` | GKS AC2: two samples per edge, better quality |
| `act(G, rng, trees)` | each eliminated clique replaced by a random spanning tree |
| `exact(G)` | exact Cholesky by full clique elimination (fills in) |
| `factorize(G, ..., pinned_vertex)` | keep one vertex for last (grounding SDDM matrices) |

`Float` and `Index` are template parameters (e.g. `float`, `std::uint32_t`).
The approximate factors are unbiased: `E[L L^T] = L_G`.

## Benchmark

```bash
./build/benchmark_cpp 100000     # random graph, prints AC/AC2 time in ms
```
