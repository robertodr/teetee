#include <cstdlib>

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#define H5_USE_XTENSOR
#include <highfive/H5Easy.hpp>

#include "utils.hpp"

int main() {
    H5Easy::File file("svd_bench.h5", H5Easy::File::ReadOnly);

    using xt_matrix = xt::xtensor<double, 2>;

    spdlog::set_pattern("[%Y-%m-%d %T][%^%l%$][TID: %t, PID: %P][%!@%s:%4#] %v");

    constexpr std::size_t sz = 2683044;
    // square matrix
    std::size_t n_rows = 1638;
    std::size_t n_cols = sz / n_rows;
    auto a = H5Easy::load<std::vector<double>>(file, "/raw/a");

    std::array<std::size_t, 2> shape = {n_rows, n_cols};
    xt_matrix A = xt::adapt(a, shape);

    SPDLOG_INFO("size of matrix = {} elements", A.size());
    SPDLOG_INFO("memory ~ {:.2e} GiB", to_GiB<double>(A.size()));

    auto start = std::chrono::steady_clock::now();

    auto [U_A, s_A, Vt_A] = xt::linalg::svd(A, false, true);

    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;

    SPDLOG_INFO("xtensor :: SVD of {}x{} square matrix done in {}", A.shape(0), A.shape(1), elapsed);

    auto ref_s_A = H5Easy::load<std::vector<double>>(file, "/ref/s_A");
    auto good = true;
    for (auto i = 0; i < ref_s_A.size(); ++i) {
        if (std::abs((ref_s_A[i] - s_A[i]) / ref_s_A[i]) >= 1.0e-12) {
            SPDLOG_INFO("std::abs((ref_s_A[{0}] - s_A[{0}]) / ref_s_A[{0}]) = {1}", i, std::abs((ref_s_A[i] - s_A[i]) / ref_s_A[i]));
            good = false;
        }
    }
    if (!good) { SPDLOG_WARN("reference and xtensor SVDs for A disagree..."); }

    // broad rectangular matrix
    n_rows = 546;
    n_cols = sz / n_rows;
    auto b = H5Easy::load<std::vector<double>>(file, "/raw/b");

    shape = {n_rows, n_cols};
    xt_matrix B = xt::adapt(b, shape);

    SPDLOG_INFO("size of matrix = {} elements", B.size());
    SPDLOG_INFO("memory ~ {:.2e} GiB", to_GiB<double>(B.size()));

    start = std::chrono::steady_clock::now();

    auto [U_B, s_B, Vt_B] = xt::linalg::svd(B, false, true);

    stop = std::chrono::steady_clock::now();
    elapsed = stop - start;

    SPDLOG_INFO("xtensor :: SVD of {}x{} broad rectangular matrix done in {}", B.shape(0), B.shape(1), elapsed);

    auto ref_s_B = H5Easy::load<std::vector<double>>(file, "/ref/s_B");
    good = true;
    for (auto i = 0; i < ref_s_B.size(); ++i) {
        if (std::abs((ref_s_B[i] - s_B[i]) / ref_s_B[i]) >= 1.0e-12) {
            SPDLOG_INFO("std::abs((ref_s_B[{0}] - s_B[{0}]) / ref_s_B[{0}]) = {1}", i, std::abs((ref_s_B[i] - s_B[i]) / ref_s_B[i]));
            good = false;
        }
    }
    if (!good) { SPDLOG_WARN("reference and xtensor SVDs for B disagree..."); }

    // skinny rectangular matrix (swap dimensions of the broad matrix)
    auto c = H5Easy::load<std::vector<double>>(file, "/raw/c");

    shape = {n_cols, n_rows};
    xt_matrix C = xt::adapt(c, shape);

    SPDLOG_INFO("size of matrix = {} elements", C.size());
    SPDLOG_INFO("memory ~ {:.2e} GiB", to_GiB<double>(C.size()));

    start = std::chrono::steady_clock::now();

    auto [U_C, s_C, Vt_C] = xt::linalg::svd(C, false, true);

    stop = std::chrono::steady_clock::now();
    elapsed = stop - start;

    SPDLOG_INFO("xtensor :: SVD of {}x{} skinny rectangular matrix done in {}", C.shape(0), C.shape(1), elapsed);

    auto ref_s_C = H5Easy::load<std::vector<double>>(file, "/ref/s_C");
    good = true;
    for (auto i = 0; i < ref_s_C.size(); ++i) {
        if (std::abs((ref_s_C[i] - s_C[i]) / ref_s_C[i]) >= 1.0e-12) {
            SPDLOG_INFO("std::abs((ref_s_C[{0}] - s_C[{0}]) / ref_s_C[{0}]) = {1}", i, std::abs((ref_s_C[i] - s_C[i]) / ref_s_C[i]));
            good = false;
        }
    }
    if (!good) { SPDLOG_WARN("reference and xtensor SVDs for C disagree..."); }

    return EXIT_SUCCESS;
}
