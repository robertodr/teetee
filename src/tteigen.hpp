#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <unsupported/Eigen/CXX11/Tensor>

#include "utils.hpp"

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

template <typename T, int D> TensorTrain<T, D> tt_svd(const Eigen::Tensor<T, D> &A, double epsilon = 1e-12) {
    std::cout << "size of tensor = " << A.size() << " elements\nmemory ~ " << to_GiB<T>(A.size()) << " GiB" << std::endl;

    // norm of tensor --> gives us the threshold for the SVDs
    const Eigen::Tensor<T, 0> A_norm = A.square().sum().sqrt();
    const double A_F = A_norm.coeff();
    std::cout << "tensor norm = " << A_F << std::endl;

    // SVD threshold
    const auto delta = (epsilon * A_F) / std::sqrt(D - 1);
    std::cout << "SVD threshold = " << delta << std::endl;
    // blocked divide-and-conquer SVD object
    // FIXME the implementation should switch between JacobiSVD for smaller matrices and BDCSVD for larger ones
    Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd;
    svd.setThreshold(delta);

    // outputs from TT-SVD
    TensorTrain<T, D> tt;
    // set "border" ranks to 1
    tt.ranks.front() = 1;
    tt.ranks.back() = 1;
    // dimensions of each mode
    for (auto i = 0; i < D; ++i) { tt.modes[i] = A.dimension(i); }

    // initialize:
    // 1. Copy input tensor to temporary B
    // we only do this to keep the original data around for the final test
    Eigen::Tensor<T, D> B(A);
    // 2. Prepare first unfolding
    auto n_rows = A.dimension(0);
    auto n_cols = static_cast<std::size_t>(A.size() / n_rows);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> M(B.data(), n_rows, n_cols);
    // 3. Compute SVD of unfolding
    std::cout << ">-> decomposing mode 0" << std::endl;
    // NOTE one needs to truncate the results of the SVD to the revealed rank (otherwise we're not actually compressing anything!)
    auto M_svd = svd.compute(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::cout << ">-> SVD for mode 0 done" << std::endl;
    // 4. Define ranks and cores
    auto rank = M_svd.rank();
    std::cout << "rank " << rank << std::endl;
    tt.ranks[1] = rank;
    // only take the first r columns of U
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> U = M_svd.matrixU()(Eigen::all, Eigen::seqN(0, rank));
    // fill tt.cores[0]
    tt.cores[0] = Eigen::Tensor<T, 3>(tt.ranks[0], A.dimension(0), rank);
    std::copy(U.data(), U.data() + tt.cores[0].size(), tt.cores[0].data());
    // 5. Next: only use first r singular values and first r columns of V
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> next = M_svd.singularValues().head(rank).asDiagonal() * M_svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).transpose();

    // go through the modes (dimensions) in the tensor
    for (int K = 1; K < D - 1; ++K) {
        std::cout << ">-> decomposing mode " << K << std::endl;
        // 1. Redefine sizes
        n_rows = A.dimension(K);
        n_cols /= n_rows;
        // 2. Construct unfolding
        new (&M) Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(next.data(), tt.ranks[K] * n_rows, n_cols);
        // 3. Compute SVD of unfolding
        M_svd = svd.compute(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        std::cout << ">-> SVD for mode " << K << " done" << std::endl;
        // 4. Define ranks and cores
        rank = M_svd.rank();
        tt.ranks[K + 1] = rank;
        // only take the first r columns of U
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> U = M_svd.matrixU()(Eigen::all, Eigen::seqN(0, rank));
        // fill tt.cores[K]
        tt.cores[K] = Eigen::Tensor<double, 3>(tt.ranks[K], n_rows, rank);
        std::copy(U.data(), U.data() + tt.cores[K].size(), tt.cores[K].data());
        // 5. Next: only use first r singular values and first r columns of V
        next = M_svd.singularValues().head(rank).asDiagonal() * M_svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).transpose();
    }
    std::cout << ">-> decomposing mode " << D - 1 << std::endl;
    // fill tt.cores[d-1]
    tt.cores[D - 1] = Eigen::Tensor<T, 3>(tt.ranks[D - 1], A.dimension(D - 1), 1);
    std::copy(next.data(), next.data() + tt.cores[D - 1].size(), tt.cores[D - 1].data());

    auto ncore = 0;
    for (const auto &c : tt.cores) {
        std::cout << "core " << ncore << std::endl;
        std::cout << "shape = [" << c.dimension(0) << ", " << c.dimension(1) << ", " << c.dimension(2) << "]" << std::endl;
        ncore += 1;
    }
    std::cout << "ranks " << tt.ranks << std::endl;
    std::cout << "size of TT format = " << tt.size() << " elements\nmemory ~ " << to_GiB<T>(tt.size()) << " GiB" << std::endl;
    std::cout << "norm of TT format = " << tt.norm() << std::endl;
    std::cout << "compression = " << (1.0 - static_cast<T>(tt.size()) / static_cast<T>(A.size())) * 100 << "%" << std::endl;

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

    // compute cores
    Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> cdims = {Eigen::IndexPair<Eigen::Index>(1, 1)};
    for (auto i = 0; i < D; ++i) {
        Eigen::Tensor<T, 3> foo = left.cores[i].contract(right.cores[i], cdims);
        std::cout << "contraction  " << foo.dimensions() << std::endl;
        retval.cores[i] = left.cores[i].contract(right.cores[i], cdims);
        // Eigen::Tensor<V, 3>(retval.ranks[i], retval.modes[i], retval.ranks[i + 1]).setZero();
    }

    return retval;
}
} // namespace tteigen
