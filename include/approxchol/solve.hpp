/// @file approxchol/solve.hpp  x = P^T (L L^T)^{-1} P b by forward and back substitution.
#pragma once
#include "approxchol/types.hpp"
namespace approxchol {
template <typename Float, std::integral Index>
std::vector<Float> solve(const cholesky_factor<Float, Index> &F, const std::vector<Float> &b) {
    const std::size_t n = F.order.size();
    std::vector<Float> y(n), x(n);
    for (std::size_t k = 0; k < n; ++k) y[k] = b[F.order[k]];
    for (std::size_t j = 0; j < n; ++j) {
        const auto &c = F.columns[j];
        if (c.empty()) continue;
        y[j] /= c[0].value;
        for (std::size_t e = 1; e < c.size(); ++e) y[c[e].row] -= c[e].value * y[j];
    }
    for (std::size_t j = n; j-- > 0;) {
        const auto &c = F.columns[j];
        if (c.empty()) continue;
        for (std::size_t e = 1; e < c.size(); ++e) y[j] -= c[e].value * y[c[e].row];
        y[j] /= c[0].value;
    }
    for (std::size_t k = 0; k < n; ++k) x[F.order[k]] = y[k];
    return x;
}
} // namespace approxchol
