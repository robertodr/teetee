#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <type_traits>

#include <fmt/chrono.h>
#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tteigen {
template <typename T, std::size_t D> struct TensorTrain final {
    using core_t = Eigen::Tensor<T, 3>;
    using size_type = Eigen::Index;

    std::array<size_type, D> modes;
    std::array<size_type, D + 1> ranks;
    // each of the cores has shape {R_{n-1}, I_{n}, R_{n}}
    // with R_{0} = 1 = R_{N} and I_{n} the size of mode n
    std::array<core_t, D> cores;

    auto size() const -> Eigen::Index {
        auto n = 0;
        for (const auto &c : cores) n += c.size();
        return n;
    }

    // TODO
    auto norm() const -> T {
        auto norm = T{0};
        return norm;
    }
};

template <typename T, int D> TensorTrain<T, D> tt_svd(Eigen::Tensor<T, D> &A, double epsilon = 1e-12) {
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    // norm of tensor --> gives us the threshold for the SVDs
    const Eigen::Tensor<T, 0> A_norm = A.square().sum().sqrt();
    const double A_F = A_norm.coeff();

    // outputs from TT-SVD
    TensorTrain<T, D> tt;
    // set "border" ranks to 1
    tt.ranks.front() = 1;
    tt.ranks.back() = 1;
    // dimensions of each mode
    for (auto i = 0; i < D; ++i) { tt.modes[i] = A.dimension(i); }

    // 1. Prepare first horizontal unfolding
    auto n_rows = A.dimension(0);
    auto n_cols = A.size() / n_rows;

    // set up SVD computations
    const auto delta = (epsilon * A_F) / std::sqrt(D - 1);
    SPDLOG_INFO("SVD threshold = {:6e}", delta);
    Eigen::BDCSVD<matrix_type> svd;
    svd.setThreshold(delta);

    Eigen::Map<matrix_type> M(A.data(), n_rows, n_cols);

    // 2. Compute SVD of unfolding
    auto start = std::chrono::steady_clock::now();
    // NOTE one needs to truncate the results of the SVD to the revealed rank (otherwise we're not actually compressing anything!)
    auto M_svd = svd.compute(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    SPDLOG_INFO(">-> decomposed mode {} in {}", 0, elapsed);

    // 3. Define ranks and cores
    auto rank = M_svd.rank();
    tt.ranks[1] = rank;
    // only take the first r columns of U
    matrix_type U = M_svd.matrixU()(Eigen::all, Eigen::seqN(0, rank));
    // fill tt.cores[0]
    tt.cores[0] = Eigen::Tensor<T, 3>(tt.ranks[0], A.dimension(0), rank);
    std::copy(U.data(), U.data() + tt.cores[0].size(), tt.cores[0].data());

    // 4. Next: only use first r singular values and first r columns of V
    matrix_type next = M_svd.singularValues().head(rank).asDiagonal() * M_svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).transpose();

    // go through the modes (dimensions) in the tensor
    for (int K = 1; K < D - 1; ++K) {
        // 1. Redefine sizes
        n_rows = A.dimension(K);
        n_cols /= n_rows;
        // 2. Construct unfolding
        new (&M) Eigen::Map<matrix_type>(next.data(), tt.ranks[K] * n_rows, n_cols);
        // 3. Compute SVD of unfolding
        start = std::chrono::steady_clock::now();
        M_svd = svd.compute(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        stop = std::chrono::steady_clock::now();
        elapsed = stop - start;
        SPDLOG_INFO(">-> decomposed mode {} in {}", K, elapsed);
        // 4. Define ranks and cores
        rank = M_svd.rank();
        tt.ranks[K + 1] = rank;
        // only take the first r columns of U
        matrix_type U = M_svd.matrixU()(Eigen::all, Eigen::seqN(0, rank));
        // fill tt.cores[K]
        tt.cores[K] = Eigen::Tensor<double, 3>(tt.ranks[K], n_rows, rank);
        std::copy(U.data(), U.data() + tt.cores[K].size(), tt.cores[K].data());
        // 5. Next: only use first r singular values and first r columns of V
        next = M_svd.singularValues().head(rank).asDiagonal() * M_svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).transpose();
    }
    // fill tt.cores[d-1]
    start = std::chrono::steady_clock::now();
    tt.cores[D - 1] = Eigen::Tensor<T, 3>(tt.ranks[D - 1], A.dimension(D - 1), 1);
    std::copy(next.data(), next.data() + tt.cores[D - 1].size(), tt.cores[D - 1].data());
    stop = std::chrono::steady_clock::now();
    elapsed = stop - start;
    SPDLOG_INFO(">-> decomposed mode {} in {}", D - 1, elapsed);

    return tt;
}

template <typename T, typename U, std::size_t D> TensorTrain<typename std::common_type<U, T>::type, D> operator*(U alpha, const TensorTrain<T, D> &tt) {
    TensorTrain<typename std::common_type<U, T>::type, D> retval = tt;
    retval.cores[0] = alpha * retval.cores[0];
    return retval;
}

template <typename T, typename U, std::size_t D> TensorTrain<typename std::common_type<U, T>::type, D> operator*(const TensorTrain<T, D> &tt, U alpha) {
    TensorTrain<typename std::common_type<U, T>::type, D> retval = tt;
    retval.cores[0] = alpha * retval.cores[0];
    return retval;
}

template <typename T, typename U, std::size_t D> TensorTrain<typename std::common_type<U, T>::type, D> operator+(const TensorTrain<T, D> &left, const TensorTrain<U, D> &right) {
    using V = typename std::common_type<U, T>::type;
    TensorTrain<V, D> retval;

    // left and right are congruent iff their modes array is the same
    auto congruent = true;
    for (auto i = 0; i < D; ++i) {
        if (left.modes[i] != right.modes[i]) {
            congruent = false;
            break;
        }
    }
    if (!congruent) { std::abort(); }

    // modes
    retval.modes = left.modes;

    // ranks
    retval.ranks.front() = 1;
    retval.ranks.back() = 1;
    for (auto i = 1; i < retval.ranks.size() - 1; ++i) { retval.ranks[i] = left.ranks[i] + right.ranks[i]; }

    // stack cores
    for (auto i = 0; i < D; ++i) {
        retval.cores[i] = Eigen::Tensor<V, 3>(retval.ranks[i], retval.modes[i], retval.ranks[i + 1]).setZero();
        // left operand in "upper left" corner
        Eigen::array<Eigen::Index, 3> offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> extents = {left.cores[i].dimension(0), left.cores[i].dimension(1), left.cores[i].dimension(2)};
        retval.cores[i].slice(offsets, extents) = left.cores[i];

        // right operand in "lower right" corner
        offsets = {retval.cores[i].dimension(0) - right.cores[i].dimension(0), 0, retval.cores[i].dimension(2) - right.cores[i].dimension(2)};
        extents = {right.cores[i].dimension(0), right.cores[i].dimension(1), right.cores[i].dimension(2)};
        retval.cores[i].slice(offsets, extents) = right.cores[i];
    }

    return retval;
}

template <typename T, typename U, std::size_t D> TensorTrain<typename std::common_type<U, T>::type, D> hadamard_product(const TensorTrain<T, D> &left, const TensorTrain<U, D> &right) {
    using V = typename std::common_type<U, T>::type;
    TensorTrain<V, D> retval;

    // left and right are congruent iff their modes array is the same
    auto congruent = true;
    for (auto i = 0; i < D; ++i) {
        if (left.modes[i] != right.modes[i]) {
            congruent = false;
            break;
        }
    }
    if (!congruent) { std::abort(); }

    // modes
    retval.modes = left.modes;

    // ranks
    retval.ranks.front() = 1;
    retval.ranks.back() = 1;
    for (auto i = 1; i < retval.ranks.size() - 1; ++i) { retval.ranks[i] = left.ranks[i] * right.ranks[i]; }

    // compute cores as the tensor product of the lateral slices
    Eigen::array<Eigen::IndexPair<Eigen::Index>, 0> cdims = {};
    for (auto i = 0; i < D; ++i) {
        retval.cores[i] = Eigen::Tensor<T, 3>(retval.ranks[i], retval.modes[i], retval.ranks[i + 1]);
        // loop over slices
        auto k = 0;
        auto l = 0;
        // for (auto j = 0; j < retval.modes[i]; ++j) {
        //     // define slices in result core
        //     Eigen::array<Eigen::Index, 3> offsets = {k, j, l};
        //     Eigen::array<Eigen::Index, 3> extents = {k + retval.ranks[i], j, l + retval.ranks[i + 1]};

        //    // define slices in left and right operands
        //    retval.cores[i].slice(offsets, extents) = left.cores[i].slice(offsets, extents).contract(right.cores[i].slice(offsets, extents), cdims);
        //}
        // std::cout << "retval.cores[" << i << "]\n" << retval.cores[i] << std::endl;
        SPDLOG_INFO("retval.cores[{}].dimensions()\n{}\n", i, retval.cores[i].dimensions());
        k += 1;
        l += 1;
    }

    return retval;
}

/** mode-k unfolding of a D-mode tensor. */
template <typename T, int D> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> unfold(int mode, Eigen::Tensor<T, D> &A) {
    if (mode >= D) {
        // cannot unfold on non-existing mode!
        std::abort();
    }

    const auto n_rows = A.dimension(mode);
    const auto n_cols = A.size() / n_rows;

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(A.data(), n_rows, n_cols);
}

/** Horizontal unfolding of D-mode tensor */
template <typename T, int D> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> H_unfold(Eigen::Tensor<T, D> &A) {
    return unfold(0, A);
}

/** Vertical unfolding of D-mode tensor */
template <typename T, int D> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V_unfold(Eigen::Tensor<T, D> &A) {
    return unfold(D - 1, A);
}
} // namespace tteigen
