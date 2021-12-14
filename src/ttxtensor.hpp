#pragma once

#include <array>
#include <chrono>
#include <type_traits>

#include <fmt/chrono.h>
#include <spdlog/spdlog.h>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xtensor.hpp>

namespace ttxt {
template <typename T, std::size_t D> struct TensorTrain final {
    using size_type = std::size_t;

    using vector_type = xt::xtensor<T, 1>;
    using matrix_type = xt::xtensor<T, 2>;
    using core_type = xt::xtensor<T, 3>;

    std::array<size_type, D> modes;
    std::array<size_type, D + 1> ranks;
    // each of the cores has shape {R_{n-1}, I_{n}, R_{n}}
    // with R_{0} = 1 = R_{N} and I_{n} the size of mode n
    std::array<core_type, D> cores;

    auto size() const -> size_type {
        auto n = 0;
        for (const auto &c : cores) n += c.size();
        return n;
    }
};

template <typename T, std::size_t D> TensorTrain<T, D> tt_svd(xt::xtensor<T, D> &A, double epsilon = 1e-12) {
    using vector_type = typename TensorTrain<T, D>::vector_type;
    using matrix_type = typename TensorTrain<T, D>::matrix_type;
    using core_type = typename TensorTrain<T, D>::core_type;

    // norm of tensor --> gives us the threshold for the SVDs
    const auto A_F = xt::norm_l2(A)();

    // SVD threshold
    const auto delta = (epsilon * A_F) / std::sqrt(D - 1);
    SPDLOG_INFO("SVD threshold = {:6e}", delta);

    // outputs from TT-SVD
    TensorTrain<T, D> tt;
    // set "border" ranks to 1
    tt.ranks.front() = 1;
    tt.ranks.back() = 1;
    // dimensions of each mode
    for (auto i = 0; i < D; ++i) { tt.modes[i] = A.shape(i); }

    // 1. Prepare first horizontal unfolding
    auto n_rows = A.shape(0);
    auto n_cols = static_cast<std::size_t>(A.size() / n_rows);
    SPDLOG_INFO("n_rows {}, n_cols {}", n_rows, n_cols);
    auto M = xt::reshape_view(A, {n_rows, n_cols});
    SPDLOG_INFO("M.shape() {}", M.shape());
    // 3. Compute SVD of unfolding
    auto start = std::chrono::steady_clock::now();
    // NOTE one needs to truncate the results of the SVD to the revealed rank (otherwise we're not actually compressing anything!)
    auto [U, s, Vt] = xt::linalg::svd(M, false, true);
    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    SPDLOG_INFO(">-> decomposed mode {} in {}", 0, elapsed);
    // only take singular values above threshold
    s = xt::filter(s, s >= delta);
    SPDLOG_INFO("s.shape() {}", s.shape());
    SPDLOG_INFO("s {}", s);
    // 4. Define ranks and cores
    std::size_t rank = s.size();
    SPDLOG_INFO("rank {}", rank);
    tt.ranks[1] = rank;
    // only take the first rank columns of U
    // reshaping not necessary
    // U = xt::view(U, xt::all(), xt::range(0, rank));
    // SPDLOG_INFO("U.shape() {}", U.shape());
    // SPDLOG_INFO("U {}", U);
    // fill tt.cores[0]
    std::array<std::size_t, 3> shape = {1, A.shape(0), rank};
    tt.cores[0] = xt::adapt(U.data(), A.shape(0) * rank, xt::acquire_ownership(), shape);

    // 5. Next: only use first r singular values and first r rows of Vt
    matrix_type next = xt::linalg::dot(xt::diag(s), xt::eval(xt::view(Vt, xt::range(0, rank), xt::all())));

    // go through the modes (dimensions) in the tensor
    for (int K = 1; K < D - 1; ++K) {
        // 1. Redefine sizes
        n_rows = A.shape(K);
        n_cols /= n_rows;
        SPDLOG_INFO("n_rows {} n_cols {}", n_rows, n_cols);
        // 2. Construct unfolding
        matrix_type M = next.reshape({tt.ranks[K] * n_rows, n_cols});
        SPDLOG_INFO("M.shape() {}", M.shape());
        // 3. Compute SVD of unfolding
        start = std::chrono::steady_clock::now();
        auto [U, s, Vt] = xt::linalg::svd(M, false, true);
        stop = std::chrono::steady_clock::now();
        elapsed = stop - start;
        SPDLOG_INFO(">-> decomposed mode {} in {}", K, elapsed);

        // 4. Define ranks and cores
        // only take singular values above threshold
        s = xt::filter(s, s >= delta);
        SPDLOG_INFO("s.shape() {}", s.shape());
        SPDLOG_INFO("s {}", s);
        // 4. Define ranks and cores
        rank = s.size();
        tt.ranks[K + 1] = rank;
        SPDLOG_INFO("rank {}", rank);
        // fill tt.cores[K] by taking only the first rank columns of U
        shape = {tt.ranks[K], A.shape(K), rank};
        tt.cores[K] = xt::adapt(U.data(), tt.ranks[K] * A.shape(K) * rank, xt::acquire_ownership(), shape);
        // 5. Next: only use first rank singular values and first rank columns of V
        matrix_type next = xt::linalg::dot(xt::diag(s), xt::eval(xt::view(Vt, xt::range(0, rank), xt::all())));
    }
    // fill tt.cores[d-1]
    start = std::chrono::steady_clock::now();
    shape = {tt.ranks[D - 1], A.shape(D - 1), 1};
    tt.cores[D - 1] = xt::adapt(next.data(), tt.ranks[D - 1] * A.shape(D - 1), xt::acquire_ownership(), shape);
    stop = std::chrono::steady_clock::now();
    elapsed = stop - start;
    SPDLOG_INFO(">-> decomposed mode {} in {}", D - 1, elapsed);

    return tt;
}

// template <typename T, typename U, std::size_t D> TensorTrain<typename std::common_type<U, T>::type, D> operator*(U alpha, const TensorTrain<T, D> &tt) {
// TensorTrain<typename std::common_type<U, T>::type, D> retval = tt;
// retval.cores[0] = alpha * retval.cores[0];
// return retval;
// }
//
// template <typename T, typename U, std::size_t D> TensorTrain<typename std::common_type<U, T>::type, D> operator*(const TensorTrain<T, D> &tt, U alpha) {
// TensorTrain<typename std::common_type<U, T>::type, D> retval = tt;
// retval.cores[0] = alpha * retval.cores[0];
// return retval;
// }
//
// template <typename T, typename U, std::size_t D> TensorTrain<typename std::common_type<U, T>::type, D> operator+(const TensorTrain<T, D> &left, const TensorTrain<U, D> &right) {
// using V = typename std::common_type<U, T>::type;
// TensorTrain<V, D> retval;
//
//// left and right are congruent iff their modes array is the same
// auto congruent = true;
// for (auto i = 0; i < D; ++i) {
//     if (left.modes[i] != right.modes[i]) {
//         congruent = false;
//         break;
//     }
// }
// if (!congruent) { std::abort(); }
//
//// modes
// retval.modes = left.modes;
//
//// ranks
// retval.ranks.front() = 1;
// retval.ranks.back() = 1;
// for (auto i = 1; i < retval.ranks.size() - 1; ++i) { retval.ranks[i] = left.ranks[i] + right.ranks[i]; }
//
//// stack cores
// for (auto i = 0; i < D; ++i) {
//     retval.cores[i] = Eigen::Tensor<V, 3>(retval.ranks[i], retval.modes[i], retval.ranks[i + 1]).setZero();
//     // left operand in "upper left" corner
//     Eigen::array<Eigen::Index, 3> offsets = {0, 0, 0};
//     Eigen::array<Eigen::Index, 3> extents = {left.cores[i].dimension(0), left.cores[i].dimension(1), left.cores[i].dimension(2)};
//     retval.cores[i].slice(offsets, extents) = left.cores[i];
//
//     // right operand in "lower right" corner
//     offsets = {retval.cores[i].dimension(0) - right.cores[i].dimension(0), 0, retval.cores[i].dimension(2) - right.cores[i].dimension(2)};
//     extents = {right.cores[i].dimension(0), right.cores[i].dimension(1), right.cores[i].dimension(2)};
//     retval.cores[i].slice(offsets, extents) = right.cores[i];
// }
//
// return retval;
// }
//
// template <typename T, typename U, std::size_t D> TensorTrain<typename std::common_type<U, T>::type, D> hadamard_product(const TensorTrain<T, D> &left, const TensorTrain<U, D> &right) {
// using V = typename std::common_type<U, T>::type;
// TensorTrain<V, D> retval;
//
//// left and right are congruent iff their modes array is the same
// auto congruent = true;
// for (auto i = 0; i < D; ++i) {
//     if (left.modes[i] != right.modes[i]) {
//         congruent = false;
//         break;
//     }
// }
// if (!congruent) { std::abort(); }
//
//// modes
// retval.modes = left.modes;
//
//// ranks
// retval.ranks.front() = 1;
// retval.ranks.back() = 1;
// for (auto i = 1; i < retval.ranks.size() - 1; ++i) { retval.ranks[i] = left.ranks[i] * right.ranks[i]; }
//
//// compute cores as the tensor product of the lateral slices
// Eigen::array<Eigen::IndexPair<Eigen::Index>, 0> cdims = {};
// for (auto i = 0; i < D; ++i) {
//     retval.cores[i] = Eigen::Tensor<T, 3>(retval.ranks[i], retval.modes[i], retval.ranks[i + 1]);
//     // loop over slices
//     auto k = 0;
//     auto l = 0;
//     // for (auto j = 0; j < retval.modes[i]; ++j) {
//     //     // define slices in result core
//     //     Eigen::array<Eigen::Index, 3> offsets = {k, j, l};
//     //     Eigen::array<Eigen::Index, 3> extents = {k + retval.ranks[i], j, l + retval.ranks[i + 1]};
//
//     //    // define slices in left and right operands
//     //    retval.cores[i].slice(offsets, extents) = left.cores[i].slice(offsets, extents).contract(right.cores[i].slice(offsets, extents), cdims);
//     //}
//     // std::cout << "retval.cores[" << i << "]\n" << retval.cores[i] << std::endl;
//     SPDLOG_INFO("retval.cores[{}].dimensions()\n{}\n", i, retval.cores[i].dimensions());
//     k += 1;
//     l += 1;
// }
//
// return retval;
// }
//
///** mode-k unfolding of a D-mode tensor. */
// template <typename T, int D> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> unfold(int mode, Eigen::Tensor<T, D> &A) {
//     if (mode >= D) {
//         // cannot unfold on non-existing mode!
//         std::abort();
//     }
//
//     const auto n_rows = A.dimension(mode);
//     const auto n_cols = A.size() / n_rows;
//
//     return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(A.data(), n_rows, n_cols);
// }
//
///** Horizontal unfolding of D-mode tensor */
// template <typename T, int D> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> H_unfold(Eigen::Tensor<T, D> &A) {
//     return unfold(0, A);
// }
//
///** Vertical unfolding of D-mode tensor */
// template <typename T, int D> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V_unfold(Eigen::Tensor<T, D> &A) {
//     return unfold(D - 1, A);
// }
} // namespace ttxt
