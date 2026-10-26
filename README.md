# approxchol

Header-only C++20 randomized approximate Cholesky for graph Laplacians, in under
200 lines. Use the factor as a preconditioner (e.g. for CG). No dependencies.
For an SDD M-matrix, add one ground vertex joined to each row by its diagonal excess.

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

## Bring your own graph

Any type satisfying `approxchol::weighted_graph` works directly: a random-access
range of neighbor lists whose edges have `.to` and `.weight`.

```cpp
template <class E>
concept weighted_edge = requires(const E &e) {
    { e.to } -> std::convertible_to<std::size_t>;
    { e.weight } -> std::convertible_to<double>;
};

template <class G>
concept weighted_graph =
    std::ranges::random_access_range<G> && std::ranges::sized_range<G> &&
    std::ranges::input_range<std::ranges::range_reference_t<G>> &&
    weighted_edge<std::ranges::range_value_t<std::ranges::range_reference_t<G>>>;
```

```cpp
struct my_edge { int to; float weight; };
std::vector<std::vector<my_edge>> g = /* both directions of every edge */;

auto F = approxchol::ac2(g, rng);   // converted once, then factorized
```

`to_adjacency_list(g)` does the conversion explicitly.

## Factorizations

| Call | What |
| --- | --- |
| `ac1(G, rng)` | GKS AC: one clique sample per edge |
| `ac2(G, rng)` | GKS AC2: two samples per edge, better quality |
| `exact(G)` | exact Cholesky by full clique elimination (fills in) |

`Float` and `Index` are template parameters (e.g. `float`, `std::uint32_t`).
The approximate factors are unbiased: `E[L L^T] = L_G`.

## Benchmark

```bash
./build/benchmark_cpp 100000     # random graph, prints AC/AC2 time in ms
```
