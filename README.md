# approxchol

Header-only C++20 implementation of randomized approximate Cholesky factorization
for graph Laplacians (Gao, Kyng, and Spielman, [arXiv:2303.00709](https://arxiv.org/abs/2303.00709)).
The factor is intended as a preconditioner for conjugate gradients.

## Usage

```cpp
#include "approxchol/approxchol.hpp"

approxchol::adjacency_list<double> G(n);
approxchol::add_edge(G, u, v, w);

approxchol::rng64 rng(42);
auto F = approxchol::ac2(G, rng);   // P L_G P^T ≈ L L^T
auto x = approxchol::solve(F, b);   // x = P^T (L L^T)^{-1} P b
```

`ac1` and `ac2` draw one and two samples per edge; `exact` returns the exact factor.
Any type satisfying `approxchol::weighted_graph`, a random-access range of neighbor
lists whose elements expose `to` and `weight`, may be passed in place of `adjacency_list`.
An SDD M-matrix is handled by adding a ground vertex joined to each row by its
diagonal excess.

## Build

```sh
cmake -S . -B build && cmake --build build && ./build/test_approxchol
```
