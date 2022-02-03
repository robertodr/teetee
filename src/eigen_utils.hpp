#pragma once

#include <random>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <unsupported/Eigen/CXX11/Tensor>

#include "utils.hpp"

// TODO refactor:
// - sample_tensor should return the discretization of a function (passed as
// parameter)
//   over a cubic grid in D-dimensions.
// - grid funciton produces linear index and value of function.

// FIXME alignment!
namespace tteigen {
template <std::size_t... Ds,
          template <std::size_t> class Indexing = ColMajor,
          std::size_t N = sizeof...(Ds)>
auto sample_tensor() -> Eigen::Tensor<double, N> {
    auto count = (1 * ... * Ds);

    auto dimensions = std::array<std::size_t, N>{Ds...};
    Indexing<N> linear_id(dimensions);

    auto buffer = static_cast<double *>(
        std::aligned_alloc(alignof(double), sizeof(double) * count));

    // FIXME generalize! this is only really valid for 6-mode tensors!
    // fill
    for (std::size_t i0 = 0; i0 < dimensions[0]; ++i0) {
        for (std::size_t i1 = 0; i1 < dimensions[1]; ++i1) {
            for (std::size_t i2 = 0; i2 < dimensions[2]; ++i2) {
                for (std::size_t i3 = 0; i3 < dimensions[3]; ++i3) {
                    for (std::size_t i4 = 0; i4 < dimensions[4]; ++i4) {
                        for (std::size_t i5 = 0; i5 < dimensions[5]; ++i5) {
                            auto idx = linear_id({i0, i1, i2, i3, i4, i5});
                            buffer[idx] = value(i0, i1, i2, i3, i4, i5);
                        }
                    }
                }
            }
        }
    }

    Eigen::TensorMap<Eigen::Tensor<double, N>> A(buffer, dimensions);

    return A;
}

template <typename Derived>
auto randomized_svd(const Eigen::MatrixBase<Derived> &A,
                    size_t rank,
                    size_t n_oversamples = 0)
    -> std::tuple<
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
        Eigen::Vector<typename Derived::Scalar, Eigen::Dynamic>,
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>> {

    using T = typename Derived::Scalar;
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_type = Eigen::Vector<T, Eigen::Dynamic>;

    auto n_samples = (n_oversamples > 0) ? rank + n_oversamples : 2 * rank;

    // stage A: find approximate range of X

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<T> d{T{0}, T{1}};
    auto normal = [&]() { return d(gen); };
    matrix_type O = matrix_type::NullaryExpr(A.cols(), n_samples, normal);
    matrix_type Y = A * O;

    // orthonormalize
    Eigen::HouseholderQR<Eigen::Ref<matrix_type>> qr(Y);
    auto hh = qr.householderQ();
    matrix_type Q = matrix_type::Identity(Y.rows(), Y.rows());
    Q.applyOnTheLeft(hh);

    // stage B: SVD

    matrix_type B = Q.adjoint() * A;
    auto svd = B.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

    if (svd.info() != Eigen::Success) {
        SPDLOG_ERROR("SVD decomposition did not succeed!");
        std::abort();
    }

    matrix_type U = (Q * svd.matrixU())(Eigen::all, Eigen::seqN(0, rank));
    vector_type Sigma = svd.singularValues().head(rank);
    matrix_type V = svd.matrixV()(Eigen::all, Eigen::seqN(0, rank));

    return {U, Sigma, V};
}
} // namespace tteigen
