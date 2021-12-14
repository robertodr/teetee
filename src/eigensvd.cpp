#include <algorithm>
#include <cstdlib>
#include <random>

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <highfive/H5Easy.hpp>

#include "utils.hpp"

int main() {
    H5Easy::File file("svd_bench.h5", H5Easy::File::ReadOnly);

    spdlog::set_pattern("[%Y-%m-%d %T][%^%l%$][TID: %t, PID: %P][%!@%s:%4#] %v");

    // initialize Eigen blocked divide-and-conquer SVD
    Eigen::BDCSVD<Eigen::MatrixXd> svd;

    constexpr auto sz = 2683044;
    // square matrix
    auto n_rows = 1638;
    auto n_cols = sz / n_rows;
    auto a = H5Easy::load<std::vector<double>>(file, "/raw/a");

    Eigen::Map<Eigen::MatrixXd> A(a.data(), n_rows, n_cols);

    SPDLOG_INFO("size of matrix = {} elements", A.size());
    SPDLOG_INFO("memory ~ {:.2e} GiB", to_GiB<double>(A.size()));

    auto start = std::chrono::steady_clock::now();

    auto A_svd = svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;

    SPDLOG_INFO("Eigen :: SVD of {}x{} square matrix done in {}", A.rows(), A.cols(), elapsed);

    auto s_A = A_svd.singularValues();
    auto ref_s_A = H5Easy::load<std::vector<double>>(file, "/ref/s_A");

    auto good = true;
    for (auto i = 0; i < ref_s_A.size(); ++i) {
        if (std::abs((ref_s_A[i] - s_A[i]) / ref_s_A[i]) >= 1.0e-12) {
            SPDLOG_INFO("std::abs((ref_s_A[{0}] - s_A[{0}]) / ref_s_A[{0}]) = {1}", i, std::abs((ref_s_A[i] - s_A[i]) / ref_s_A[i]));
            good = false;
        }
    }
    if (!good) { SPDLOG_WARN("reference and Eigen SVDs for A disagree..."); }

    // broad rectangular matrix
    n_rows = 546;
    n_cols = sz / n_rows;
    auto b = H5Easy::load<std::vector<double>>(file, "/raw/b");

    Eigen::Map<Eigen::MatrixXd> B(b.data(), n_rows, n_cols);

    SPDLOG_INFO("size of matrix = {} elements", B.size());
    SPDLOG_INFO("memory ~ {:.2e} GiB", to_GiB<double>(B.size()));

    start = std::chrono::steady_clock::now();

    auto B_svd = svd.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    stop = std::chrono::steady_clock::now();
    elapsed = stop - start;

    SPDLOG_INFO("Eigen :: SVD of {}x{} broad rectangular matrix done in {}", B.rows(), B.cols(), elapsed);

    auto s_B = B_svd.singularValues();
    auto ref_s_B = H5Easy::load<std::vector<double>>(file, "/ref/s_B");

    good = true;
    for (auto i = 0; i < ref_s_B.size(); ++i) {
        if (std::abs((ref_s_B[i] - s_B[i]) / ref_s_B[i]) >= 1.0e-12) {
            SPDLOG_INFO("std::abs((ref_s_B[{0}] - s_B[{0}]) / ref_s_B[{0}]) = {1}", i, std::abs((ref_s_B[i] - s_B[i]) / ref_s_B[i]));
            good = false;
        }
    }
    if (!good) { SPDLOG_WARN("reference and Eigen SVDs for B disagree..."); }

    // skinny rectangular matrix (swap dimensions of the broad matrix)
    auto c = H5Easy::load<std::vector<double>>(file, "/raw/c");

    Eigen::Map<Eigen::MatrixXd> C(c.data(), n_cols, n_rows);

    SPDLOG_INFO("size of matrix = {} elements", C.size());
    SPDLOG_INFO("memory ~ {:.2e} GiB", to_GiB<double>(C.size()));

    start = std::chrono::steady_clock::now();

    auto C_svd = svd.compute(C, Eigen::ComputeThinU | Eigen::ComputeThinV);

    stop = std::chrono::steady_clock::now();
    elapsed = stop - start;

    SPDLOG_INFO("Eigen :: SVD of {}x{} skinny rectangular matrix done in {}", C.rows(), C.cols(), elapsed);

    auto s_C = C_svd.singularValues();
    auto ref_s_C = H5Easy::load<std::vector<double>>(file, "/ref/s_C");

    good = true;
    for (auto i = 0; i < ref_s_C.size(); ++i) {
        if (std::abs((ref_s_C[i] - s_C[i]) / ref_s_C[i]) >= 1.0e-12) {
            SPDLOG_INFO("std::abs((ref_s_C[{0}] - s_C[{0}]) / ref_s_C[{0}]) = {1}", i, std::abs((ref_s_C[i] - s_C[i]) / ref_s_C[i]));
            good = false;
        }
    }
    if (!good) { SPDLOG_WARN("reference and Eigen SVDs for C disagree..."); }

    return EXIT_SUCCESS;
}
