#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <type_traits>

#include <fmt/chrono.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tteigen {
using Clock = std::chrono::steady_clock;

enum class Orthonormal { No, Left, Right };

template <typename T, std::size_t D> struct TensorTrain final {
    /** Indexing */
    using size_type = Eigen::Index;
    /** Format for 0-th and (D-1)-th cores */
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    /** Shape of cores */
    using shape_type = std::array<size_type, 3>;
    /** Inner cores (1 to D-2) */
    using core_type = Eigen::Tensor<T, 3>;

    Orthonormal ortho{Orthonormal::No};

    T norm_{0};

    std::array<size_type, D> modes;
    std::array<size_type, D + 1> ranks;
    std::array<shape_type, D> shapes;
    // each of the cores has shape {R_{n-1}, I_{n}, R_{n}}
    // with R_{0} = 1 = R_{N} and I_{n} the size of mode n
    std::array<core_type, D> cores;

    /*\@{ Constructors */
    TensorTrain() = default;

    TensorTrain(std::array<size_type, D> Is, std::array<size_type, D + 1> Rs)
            : modes(Is)
            , ranks(Rs) {
        for (auto K = 0; K < D; ++K) {
            shapes[K] = {ranks[K], modes[K], ranks[K + 1]};
            cores[K] = core_type(shapes[K]).setZero();
        }
    }
    /*\@}*/

    auto size() const -> Eigen::Index {
        auto n = 0;
        for (const auto &c : cores) n += c.size();
        return n;
    }

    auto norm() -> T {
        // TODO it requires left or right orthonormalization, which might be a bi
        //    if (ortho == Orthonormal::No) {
        //        right_orthonormalize(cores);
        //        ortho = Orthonormal::Right;
        //    }

        Eigen::Tensor<T, 0> tmp;
        switch (ortho) {
            case Orthonormal::Left:
                tmp = cores[D - 1].square().sum().sqrt();
                norm_ = static_cast<T>(tmp.coeff());
                break;
            case Orthonormal::Right:
                tmp = cores[0].square().sum().sqrt();
                norm_ = static_cast<T>(tmp.coeff());
                break;
        }

        return norm_;
    }
};

template <typename T, int D>
TensorTrain<T, D> tt_svd(Eigen::Tensor<T, D> &A, double epsilon = 1e-12) {
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    // norm of tensor --> gives us the threshold for the SVDs
    const Eigen::Tensor<T, 0> A_norm = A.square().sum().sqrt();
    const double A_F = A_norm.coeff();

    SPDLOG_INFO("Frobenius norm {}", A_F);

    // outputs from TT-SVD
    TensorTrain<T, D> tt;
    // set "border" ranks to 1
    tt.ranks.front() = 1;
    tt.ranks.back() = 1;
    // dimensions of each mode
    for (auto i = 0; i < D; ++i) { tt.modes[i] = A.dimension(i); }

    // set up SVD computations
    const auto delta = (epsilon * A_F) / std::sqrt(D - 1);
    SPDLOG_INFO("SVD threshold = {:6e}", delta);

    Eigen::BDCSVD<matrix_type> svd;

    // prepare first horizontal unfolding
    auto n_rows = A.dimension(0);
    auto n_cols = A.size() / n_rows;

    Eigen::Map<matrix_type> M(A.data(), n_rows, n_cols);

    // compute SVD of unfolding
    auto start = Clock::now();
    auto M_svd = svd.compute(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto stop = Clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;

    SPDLOG_INFO("SVD decomposition of mode {} in {}", 0, elapsed);

    if (M_svd.info() != Eigen::Success) {
        fmt::print(
            "SVD decomposition of mode {} (out of {}) did not succeed!", 0, D);
        std::abort();
    }

    // define ranks and cores
    auto rank = (M_svd.singularValues().array() >= delta).count();
    tt.ranks[1] = rank;
    // only take the first rank columns of U to fill tt.cores[0]
    tt.cores[0] = Eigen::Tensor<T, 3>(tt.ranks[0], A.dimension(0), rank);
    std::copy(M_svd.matrixU().data(),
              M_svd.matrixU().data() + tt.cores[0].size(),
              tt.cores[0].data());

    // prepare next unfolding: only use first rank singular values and first rank
    // columns of V
    matrix_type next = M_svd.singularValues().head(rank).asDiagonal() *
                       M_svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).transpose();

    // go through the modes (dimensions) in the tensor
    for (int K = 1; K < D - 1; ++K) {
        // sizes of tensor unfoldings
        n_rows = A.dimension(K);
        n_cols /= n_rows;
        // construct unfolding
        new (&M) Eigen::Map<matrix_type>(next.data(), tt.ranks[K] * n_rows, n_cols);
        // compute SVD of unfolding
        start = Clock::now();
        M_svd = svd.compute(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        stop = Clock::now();
        elapsed = stop - start;
        SPDLOG_INFO("SVD decomposition of mode {} in {}", K, elapsed);

        if (M_svd.info() != Eigen::Success) {
            fmt::print(
                "SVD decomposition of mode {} (out of {}) did not succeed!", K, D);
            std::abort();
        }

        // define ranks and cores
        rank = (M_svd.singularValues().array() >= delta).count();
        tt.ranks[K + 1] = rank;

        // only take the first rank columns of U to fill tt.cores[K]
        tt.cores[K] = Eigen::Tensor<double, 3>(tt.ranks[K], n_rows, rank);
        std::copy(M_svd.matrixU().data(),
                  M_svd.matrixU().data() + tt.cores[K].size(),
                  tt.cores[K].data());

        // prepare next unfolding: only use first rank singular values and first rank
        // columns of V
        next = M_svd.singularValues().head(rank).asDiagonal() *
               M_svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).transpose();
    }

    // fill tt.cores[d-1]
    start = Clock::now();
    tt.cores[D - 1] = Eigen::Tensor<T, 3>(tt.ranks[D - 1], A.dimension(D - 1), 1);
    std::copy(
        next.data(), next.data() + tt.cores[D - 1].size(), tt.cores[D - 1].data());
    stop = Clock::now();
    elapsed = stop - start;
    SPDLOG_INFO("SVD decomposition of mode {} in {}", D - 1, elapsed);

    return tt;
}

template <typename T, typename U, std::size_t D>
TensorTrain<typename std::common_type<U, T>::type, D> operator*(
    U alpha,
    const TensorTrain<T, D> &tt) {
    TensorTrain<typename std::common_type<U, T>::type, D> retval = tt;
    retval.cores[0] = alpha * retval.cores[0];
    return retval;
}

template <typename T, typename U, std::size_t D>
TensorTrain<typename std::common_type<U, T>::type, D> operator*(
    const TensorTrain<T, D> &tt,
    U alpha) {
    TensorTrain<typename std::common_type<U, T>::type, D> retval = tt;
    retval.cores[0] = alpha * retval.cores[0];
    return retval;
}

template <typename T, typename U, std::size_t D>
TensorTrain<typename std::common_type<U, T>::type, D> operator+(
    const TensorTrain<T, D> &left,
    const TensorTrain<U, D> &right) {
    using V = typename std::common_type<U, T>::type;

    TensorTrain<V, D> retval;

    using core_type = typename TensorTrain<V, D>::core_type;

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
    for (auto i = 1; i < retval.ranks.size() - 1; ++i) {
        retval.ranks[i] = left.ranks[i] + right.ranks[i];
    }

    // stack cores
    for (auto i = 0; i < D; ++i) {
        retval.cores[i] =
            core_type(retval.ranks[i], retval.modes[i], retval.ranks[i + 1])
                .setZero();
        // left operand in "upper left" corner
        Eigen::array<Eigen::Index, 3> offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> extents = {left.cores[i].dimension(0),
                                                 left.cores[i].dimension(1),
                                                 left.cores[i].dimension(2)};
        retval.cores[i].slice(offsets, extents) = left.cores[i];

        // right operand in "lower right" corner
        offsets = {retval.cores[i].dimension(0) - right.cores[i].dimension(0),
                   0,
                   retval.cores[i].dimension(2) - right.cores[i].dimension(2)};
        extents = {right.cores[i].dimension(0),
                   right.cores[i].dimension(1),
                   right.cores[i].dimension(2)};
        retval.cores[i].slice(offsets, extents) = right.cores[i];
    }

    return retval;
}

template <typename T, typename U, std::size_t D>
TensorTrain<typename std::common_type<U, T>::type, D> hadamard_product(
    const TensorTrain<T, D> &left,
    const TensorTrain<U, D> &right) {
    using V = typename std::common_type<U, T>::type;

    TensorTrain<V, D> retval;

    using core_type = typename TensorTrain<V, D>::core_type;

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
    for (auto i = 0; i < retval.ranks.size(); ++i) {
        retval.ranks[i] = left.ranks[i] * right.ranks[i];
    }

    // compute cores as the tensor product of the slices
    Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> cdims = {
        Eigen::IndexPair<Eigen::Index>(1, 1)};
    //  offset of slices
    Eigen::array<Eigen::Index, 3> offs;
    // extents of slices
    Eigen::array<Eigen::Index, 3> l_ext, r_ext, ext;
    // shuffle
    Eigen::array<Eigen::Index, 4> shuffle = {0, 2, 1, 3};
    // loop over cores
    for (auto i = 0; i < D; ++i) {
        retval.cores[i] =
            core_type(retval.ranks[i], retval.modes[i], retval.ranks[i + 1])
                .setZero();
        // Eigen::Tensor<T, 4> tmp;
        // loop over slices
        for (auto j = 0; j < retval.modes[i]; ++j) {
            // offset of operands and result slices
            offs = {0, j, 0};
            // extent of slice for left operand
            l_ext = {left.ranks[i], 1, left.ranks[i + 1]};
            // extent of slice for right operand
            r_ext = {right.ranks[i], 1, right.ranks[i + 1]};
            // extent of slice for result
            ext = {retval.ranks[i], 1, retval.ranks[i + 1]};

            retval.cores[i].slice(offs, ext) =
                left.cores[i]
                    .slice(offs, l_ext)
                    .contract(right.cores[i].slice(offs, r_ext), cdims)
                    .shuffle(shuffle)
                    .reshape(ext);
        }
    }

    return retval;
}

/** Horizontal unfolding of 3-mode tensor
 *
 *  Given a tensor \f$\mathcal{T} \in \mathbb{K}^{N\times L \times M}\f$,
 *  generate a matrix \f$\mathcal{H}(\mathcal{T}) \in \mathbb{K}^{N\times LM}$
 *  by concatenating the slices \f$\mathbf{X}_{\mathcal{T}}(:, l, :) \in
 *  \mathbb{K}^{N\times M}\f$ _horizontally_.
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> horizontal_unfolding(
    Eigen::Tensor<T, 3> &A) {
    const auto n_rows = A.dimension(0);
    const auto n_cols = A.dimension(1) * A.dimension(2);

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        A.data(), n_rows, n_cols);
}

/** Vertical unfolding of 3-mode tensor
 *
 *  Given a tensor \f$\mathcal{T} \in \mathbb{K}^{N\times L \times M}\f$,
 *  generate a matrix \f$\mathcal{V}(\mathcal{T}) \in \mathbb{K}^{NL\times M}$
 *  by concatenating the slices \f$\mathbf{X}_{\mathcal{T}}(:, l, :) \in
 *  \mathbb{K}^{N\times M}\f$ _vertically_.
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> vertical_unfolding(
    Eigen::Tensor<T, 3> &A) {
    const auto n_rows = A.dimension(0) * A.dimension(1);
    const auto n_cols = A.dimension(2);

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        A.data(), n_rows, n_cols);
}

/** Right-orthonormalization of tensor train.
 *
 * @note This is non-destructive.
 * @note We only use the "thin" \f$\mathbt{Q}_{1}\f$ and \f$\mathbt{R}_{1}\f$.
 */
template <typename T, std::size_t D>
TensorTrain<T, D> right_orthonormalize(TensorTrain<T, D> &X) {
    using core_type = typename TensorTrain<T, D>::core_type;
    using matrix_type = typename TensorTrain<T, D>::matrix_type;

    TensorTrain<T, D> Y(X.modes, X.ranks);

    // start from last core and go down to second mode
    Y.cores[D - 1] = X.cores[D - 1];

    for (auto i = D - 1; i > 0; --i) {
        // why not auto? .transpose() returns an expression and hence its QR is
        // not well-defined.  To avoid this issue, we need to instantiate it as
        // a matrix.
        matrix_type Ht = horizontal_unfolding(Y.cores[i]).transpose();
        auto m = Ht.rows();
        auto n = Ht.cols();

        auto start = Clock::now();
        auto qr = Ht.householderQr();
        auto stop = Clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        SPDLOG_INFO("QR decomposition of mode {} in {}", i, elapsed);

        // get thin Q factor...
        matrix_type Q1 = Eigen::MatrixXd::Identity(m, n);
        qr.householderQ().applyThisOnTheLeft(Q1);
        // ...transpose it...
        matrix_type Q1t = Q1.transpose();
        // ...copy it to the current core
        std::copy(Q1t.data(), Q1t.data() + Y.cores[i].size(), Y.cores[i].data());

        // R factors for thin QR
        matrix_type R = qr.matrixQR()
                            .topLeftCorner(n, n)
                            .template triangularView<Eigen::Upper>();

        matrix_type next = vertical_unfolding(X.cores[i - 1]) * R.transpose();
        // ...and set the result to be the vertical unfolding of the next core of Y
        std::copy(
            next.data(), next.data() + Y.cores[i - 1].size(), Y.cores[i - 1].data());
    }

    Y.ortho = Orthonormal::Right;

    return Y;
}
} // namespace tteigen
