/// @file approxchol/solve.hpp  x = P^T (L L^T)^{-1} P b; scratch `y` is reusable across calls.
#pragma once
#include "approxchol/types.hpp"
namespace approxchol {
template <typename Float, std::integral Index>
void solve(const cholesky_factor<Float, Index> &F, const Float *b, Float *x, std::vector<Float> &y) {
    const std::size_t n = F.order.size();
    y.resize(n);
    for (std::size_t k = 0; k < n; ++k) y[k] = b[F.order[k]];
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t p = F.start[j], q = F.start[j + 1]; if (p == q) continue;
        const Float yj = (y[j] /= F.value[p]); // local: the stores below may alias y[j]
        for (std::size_t e = p + 1; e < q; ++e) y[F.row[e]] -= F.value[e] * yj;
    }
    for (std::size_t j = n; j-- > 0;) {
        const std::size_t p = F.start[j], q = F.start[j + 1]; if (p == q) continue;
        Float sum = y[j]; // local accumulator, as above
        for (std::size_t e = p + 1; e < q; ++e) sum -= F.value[e] * y[F.row[e]];
        y[j] = sum / F.value[p];
    }
    for (std::size_t k = 0; k < n; ++k) x[F.order[k]] = y[k];
}
template <typename Float, std::integral Index>
std::vector<Float> solve(const cholesky_factor<Float, Index> &F, const std::vector<Float> &b) {
    std::vector<Float> x(b.size()), y;
    return solve(F, b.data(), x.data(), y), x;
}
} // namespace approxchol
