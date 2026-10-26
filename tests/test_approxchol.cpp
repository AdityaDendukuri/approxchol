#include "approxchol/approxchol.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

template <typename Float, typename Index>
Float get_entry(const approxchol::cholesky_factor<Float, Index> &F, Index row, Index col) {
    for (std::size_t e = F.start[col]; e < F.start[col + 1]; ++e)
        if (F.row[e] == row) return F.value[e];
    return Float(0);
}

void check(bool condition, const char *msg) {
    if (!condition) {
        throw std::runtime_error(std::string("Test assertion failed: ") + msg);
    }
}

template <typename Float = double, typename Index = std::size_t>
approxchol::adjacency_list<Float, Index> make_grid_laplacian(Index rows, Index cols) {
    const Index n = rows * cols;
    approxchol::adjacency_list<Float, Index> G(n);

    auto index = [cols](Index r, Index c) { return r * cols + c; };

    for (Index r = 0; r < rows; ++r) {
        for (Index c = 0; c < cols; ++c) {
            Index u = index(r, c);
            if (c + 1 < cols) {
                approxchol::add_edge(G, u, index(r, c + 1), static_cast<Float>(1.0));
            }
            if (r + 1 < rows) {
                approxchol::add_edge(G, u, index(r + 1, c), static_cast<Float>(1.0));
            }
        }
    }

    return G;
}

template <typename Float = double, typename Index = std::size_t>
std::vector<std::vector<Float>>
dense_from_graph(const approxchol::adjacency_list<Float, Index> &G) {
    const Index n = static_cast<Index>(G.size());
    std::vector<std::vector<Float>> A(n, std::vector<Float>(n, static_cast<Float>(0)));

    for (Index u = 0; u < n; ++u) {
        Float diag = static_cast<Float>(0);
        for (const auto &e : G[u]) {
            A[u][e.to] -= e.weight;
            diag += e.weight;
        }
        A[u][u] = diag;
    }

    return A;
}

void test_exact_factorization() {
    const std::size_t rows = 4;
    const std::size_t cols = 4;
    const std::size_t n = rows * cols;

    auto G = make_grid_laplacian<double, std::size_t>(rows, cols);
    auto A_dense = dense_from_graph(G);

    auto F = approxchol::exact(G);

    std::vector<std::size_t> pos(n);
    for (std::size_t step = 0; step < n; ++step) {
        pos[F.order[step]] = step;
    }

    double max_err = 0.0;
    for (std::size_t u = 0; u < n; ++u) {
        for (std::size_t v = 0; v < n; ++v) {
            double l_lt = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                double l_uk = get_entry(F, pos[u], k);
                double l_vk = get_entry(F, pos[v], k);
                if (l_uk != 0.0 && l_vk != 0.0) {
                    l_lt += l_uk * l_vk;
                }
            }
            max_err = std::max(max_err, std::abs(l_lt - A_dense[u][v]));
        }
    }

    check(max_err < 1e-12, "Exact Cholesky error exceeds tolerance");
    std::cout << "test_exact_factorization passed (max error = " << max_err << ")\n";
}

void test_ac1_and_ac2_factorization() {
    const std::size_t rows = 4;
    const std::size_t cols = 4;
    const std::size_t n = rows * cols;

    auto G = make_grid_laplacian<double, std::size_t>(rows, cols);

    auto F1 = approxchol::ac1(G, /*seed=*/42);
    auto F2 = approxchol::ac2(G, /*seed=*/42);

    check(F1.start.size() == n + 1, "AC1 factor column count mismatch");
    check(F2.start.size() == n + 1, "AC2 factor column count mismatch");

    std::cout << "test_ac1_and_ac2_factorization passed\n";
}

void test_templated_types_float_and_uint32() {
    const std::uint32_t rows = 4;
    const std::uint32_t cols = 4;
    const std::uint32_t n = rows * cols;

    auto G = make_grid_laplacian<float, std::uint32_t>(rows, cols);

    std::mt19937 custom_rng(777);
    auto F = approxchol::ac2(G, custom_rng);

    check(F.start.size() == n + 1, "Templated factor column count mismatch");
    std::cout << "test_templated_types_float_and_uint32 passed\n";
}

void test_full_factor_expectation() {
    const std::size_t rows = 3;
    const std::size_t cols = 3;
    const std::size_t n = rows * cols;
    const std::size_t trials = 3000;

    auto G = make_grid_laplacian<double, std::size_t>(rows, cols);
    auto A_dense = dense_from_graph(G);

    std::vector<std::vector<double>> E_LLt(n, std::vector<double>(n, 0.0));
    std::mt19937_64 rng(123);

    for (std::size_t t = 0; t < trials; ++t) {
        auto F = approxchol::ac2(G, rng);

        std::vector<std::size_t> pos(n);
        for (std::size_t step = 0; step < n; ++step) {
            pos[F.order[step]] = step;
        }

        for (std::size_t u = 0; u < n; ++u) {
            std::size_t row_u = pos[u];
            for (std::size_t v = 0; v < n; ++v) {
                std::size_t row_v = pos[v];
                double val = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                    double l_uk = get_entry(F, row_u, k);
                    double l_vk = get_entry(F, row_v, k);
                    if (l_uk != 0.0 && l_vk != 0.0) {
                        val += l_uk * l_vk;
                    }
                }
                E_LLt[u][v] += val;
            }
        }
    }

    double max_mc_err = 0.0;
    for (std::size_t u = 0; u < n; ++u) {
        for (std::size_t v = 0; v < n; ++v) {
            double mean_val = E_LLt[u][v] / static_cast<double>(trials);
            max_mc_err = std::max(max_mc_err, std::abs(mean_val - A_dense[u][v]));
        }
    }

    check(max_mc_err < 0.1, "Full factor expectation error too large");
    std::cout << "test_full_factor_expectation passed (max error = " << max_mc_err << ")\n";
}

void test_triangular_solve() {
    const std::size_t rows = 4;
    const std::size_t cols = 4;
    const std::size_t n = rows * cols;

    auto G = make_grid_laplacian<double, std::size_t>(rows, cols);
    // Add ground/leak to node 0 so A is strictly positive definite
    G[0].push_back({0, 1.0, 1}); // Self-loop to make SDDM strictly SPD
    auto A_dense = dense_from_graph(G);

    auto F = approxchol::exact(G);

    std::vector<double> x_true(n);
    for (std::size_t i = 0; i < n; ++i) {
        x_true[i] = static_cast<double>(i + 1);
    }

    std::vector<double> b(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            b[i] += A_dense[i][j] * x_true[j];
        }
    }

    auto x_computed = approxchol::solve(F, b);

    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::abs(x_computed[i] - x_true[i]));
    }

    check(max_err < 1e-11, "Triangular solve error exceeds tolerance");
    std::cout << "test_triangular_solve passed (max error = " << max_err << ")\n";
}

struct user_edge {
    int to;
    float weight;
};

void test_user_graph_concept() {
    static_assert(approxchol::weighted_graph<std::vector<std::vector<user_edge>>>);
    static_assert(!approxchol::weighted_graph<std::vector<int>>);

    const std::size_t rows = 4;
    const std::size_t cols = 4;
    auto G = make_grid_laplacian<double, std::size_t>(rows, cols);

    std::vector<std::vector<user_edge>> mine(G.size());
    for (std::size_t u = 0; u < G.size(); ++u) {
        for (const auto &e : G[u]) {
            mine[u].push_back({static_cast<int>(e.to), static_cast<float>(e.weight)});
        }
    }

    auto F_mine = approxchol::exact(mine);
    auto F_ref = approxchol::exact(G);
    check(F_mine.order == F_ref.order, "User-graph exact factor ordering differs");
    check(F_mine.start == F_ref.start && F_mine.row == F_ref.row,
          "User-graph exact factor sparsity differs");
    for (std::size_t e = 0; e < F_ref.value.size(); ++e)
        check(std::abs(F_mine.value[e] - F_ref.value[e]) < 1e-12,
              "User-graph exact factor values differ");

    approxchol::rng64 rng(7);
    auto F_ac2 = approxchol::ac2(mine, rng);
    check(F_ac2.start.size() == G.size() + 1, "User-graph AC2 factor column count mismatch");
    std::cout << "test_user_graph_concept passed\n";
}

} // namespace

int main() {
    try {
        test_exact_factorization();
        test_triangular_solve();
        test_ac1_and_ac2_factorization();
        test_templated_types_float_and_uint32();
        test_full_factor_expectation();
        test_user_graph_concept();
        std::cout << "All tests passed.\n";
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
