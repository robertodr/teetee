#pragma once

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

#include "utils.hpp"

namespace tteigen {
using Clock = std::chrono::steady_clock;

namespace tteigen {
/**
 *
 *  Horizontal unfolding
 *  Given a 3-mode tensor \f$\mathcal{T} \in \mathbb{K}^{N\times L \times
 *  M}\f$, the horizontal unfolding generates a matrix
 *  \f$\mathcal{H}(\mathcal{T}) \in \mathbb{K}^{N\times LM}$ by
 *  concatenating the slices \f$\mathbf{X}_{\mathcal{T}}(:, l, :) \in
 *  \mathbb{K}^{N\times M}\f$ _horizontally_.
 *
 *  Vertical unfolding
 *  Given a 3-mode tensor \f$\mathcal{T} \in \mathbb{K}^{N\times L \times M}\f$,
 *  the vertical unfolding generates a matrix \f$\mathcal{V}(\mathcal{T}) \in
 * \mathbb{K}^{NL\times M}$ by concatenating the slices
 * \f$\mathbf{X}_{\mathcal{T}}(:, l, :) \in \mathbb{K}^{N\times M}\f$
 * _vertically_.
 */
template <typename T, int D> class TensorTrain final {
    static_assert(std::is_floating_point_v<T>,
                  "TensorTrain can only be instantiated with floating point types!");

public:
    /** Indexing */
    using size_type = Eigen::Index;
    /** Shape of cores */
    using shape_type = std::array<size_type, 3>;
    /** Cores */
    using core_type = Eigen::Tensor<T, 3>;
    /** Tensor core unfoldings */
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    /** Index pair (useful for contractions, offsets, extents) */
    using index_pair_type = Eigen::IndexPair<Eigen::Index>;
    /** Extent */
    template <int n> using extent_type = Eigen::array<index_pair_type, n>;

private:
    using Clock = std::chrono::steady_clock;

    /** Whether the tensor train is right-orthonormal. */
    bool is_orthonormal_{false};
    /** Whether we computed the norm for this tensor train. */
    bool norm_computed_{false};

    /** Number of elements in the compressed (tensor train) representation. */
    size_type c_count_{0};
    /** Number of elements in the uncompressed representation. */
    size_type u_count_{0};

    /** Norm of the tensor train. */
    T norm_{0};

    /** Decomposition threshold. */
    T epsilon_{1e-12};

    /** Sizes of tensor modes \f$I_{n}\f$ */
    std::array<size_type, D> modes_;
    /** Ranks of tensor cores \f$R_{n}\f$ */
    std::array<size_type, D + 1> ranks_;
    /** Shapes of tensor cores \f$\lbrace R_{n-1}, I_{n}, R_{n} \rbrace\f$
     * @note \f$R_{0} = 1 = R_{n}\f$
     */
    std::array<shape_type, D> shapes_;
    /** Tensor train cores: \f$\mathcal{T}_{\mathcal{X}, n} \in \mathbb{K}^{R_{n-1}
     * \times I_{n} \times R_{n}}\f$ */
    std::array<core_type, D> cores_;

    /** Tensor train decomposition *via* successive SVDs
     *  @param[in] A dense tensor data in *natural descending order*.
     *  @param[in] delta SVD truncation threshold.
     *
     * This is an implementation of algorithm 1 in: Oseledets, I. V. Tensor-Train
     * Decomposition. SIAM J. Sci. Comput. 2011, 33 (5), 2295–2317.
     * https://doi.org/10.1137/090752286.
     *
     * @note We use the block divide-and-conquer SVD algorithm, as implemented in
     * Eigen.
     */
    void decompose(T *A, T delta) {
        using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        SPDLOG_INFO("SVD threshold = {:6e}", delta);

        // prepare first horizontal unfolding
        auto n_rows = modes_[0];
        auto n_cols = u_count_ / n_rows;

        // wrap tensor data into a matrix
        Eigen::Map<matrix_type> M(A, n_rows, n_cols);

        // compute SVD of unfolding
        auto start = Clock::now();
        auto svd = M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        auto stop = Clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;

        SPDLOG_INFO("SVD decomposition of mode {} in {}", 0, elapsed);

        if (svd.info() != Eigen::Success) {
            SPDLOG_ERROR(
                "SVD decomposition of mode {} (out of {}) did not succeed!", 0, D);
            std::abort();
        }

        // define ranks and cores
        auto rank = (svd.singularValues().array() >= delta).count();
        shapes_[0] = {1, modes_[0], rank};
        ranks_[1] = rank;
        cores_[0] = core_type(shapes_[0]);
        // only take the first rank columns of U to fill cores_[0]
        std::copy(svd.matrixU().data(),
                  svd.matrixU().data() + cores_[0].size(),
                  cores_[0].data());
        c_count_ = cores_[0].size();

        // prepare next unfolding: only use first rank singular values and first rank
        // columns of V
        matrix_type next = svd.singularValues().head(rank).asDiagonal() *
                           svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).adjoint();

        // go through the modes (dimensions) in the tensor
        for (int K = 1; K < D - 1; ++K) {
            // sizes of tensor unfoldings
            n_rows = modes_[K];
            n_cols /= n_rows;
            // construct unfolding
            new (&M)
                Eigen::Map<matrix_type>(next.data(), ranks_[K] * n_rows, n_cols);
            // compute SVD of unfolding
            start = Clock::now();
            svd = M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
            stop = Clock::now();
            elapsed = stop - start;
            SPDLOG_INFO("SVD decomposition of mode {} in {}", K, elapsed);

            if (svd.info() != Eigen::Success) {
                SPDLOG_ERROR(
                    "SVD decomposition of mode {} (out of {}) did not succeed!",
                    K,
                    D);
                std::abort();
            }

            // define ranks and cores
            rank = (svd.singularValues().array() >= delta).count();
            shapes_[K] = {ranks_[K], modes_[K], rank};
            ranks_[K + 1] = rank;
            cores_[K] = core_type(shapes_[K]);
            c_count_ += cores_[K].size();
            // only take the first rank columns of U to fill cores_[K]
            std::copy(svd.matrixU().data(),
                      svd.matrixU().data() + cores_[K].size(),
                      cores_[K].data());

            // prepare next unfolding: only use first rank singular values and first
            // rank columns of V
            next = svd.singularValues().head(rank).asDiagonal() *
                   svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).adjoint();
        }

        shapes_[D - 1] = {ranks_[D - 1], modes_[D - 1], ranks_[D]};
        cores_[D - 1] = core_type(shapes_[D - 1]);
        c_count_ += cores_[D - 1].size();
        // fill cores_[D-1]
        start = Clock::now();
        std::copy(
            next.data(), next.data() + cores_[D - 1].size(), cores_[D - 1].data());
        stop = Clock::now();
        elapsed = stop - start;
        SPDLOG_INFO("SVD decomposition of mode {} in {}", D - 1, elapsed);
    }

    /*\@{ Recursive tensor train reconstruction to full format. */
    /** Reconstruct full tensor from `*this` tensor train.
     *
     * @tparam mode the index of the tensor train core that is right operand in
     * the contraction.
     * @param[in] pre result of the previous contraction, i.e. a tensor with mode+1
     * modes.
     * @param[in,out] full the full format for `*this` tensor train.
     */
    template <int mode>
    auto to_full(const Eigen::Tensor<T, mode + 1> &pre,
                 Eigen::Tensor<T, D> &full) const -> void {
        static_assert(mode >= 2, "mode must be at least two at this point!");
        extent_type<1> cdims = {index_pair_type(mode, 0)};

        Eigen::Tensor<T, mode + 2> post = pre.contract(cores_[mode], cdims);

        to_full<mode + 1>(post, full);
    }

    /** Reconstruct full tensor from `*this` tensor train. Bottom of recursion.
     *
     * @param[in] pre result of the previous contraction, i.e. a tensor with mode+1
     * modes.
     * @param[in,out] full the full format for `*this` tensor train.
     */
    template <>
    auto to_full<D - 1>(const Eigen::Tensor<T, D> &pre,
                        Eigen::Tensor<T, D> &full) const -> void {
        extent_type<1> cdims = {index_pair_type(D - 1, 0)};

        // the contraction would be an Eigen::Tensor<T, D+1>, with last mode of size
        // 1, hence the chipping.
        full = pre.contract(cores_[D - 1], cdims).chip(0, D);
    }
    /*\@}*/

public:
    /*\@{ Constructors */
    TensorTrain() = default;

    /** Zero-initialize a tensor train from given modes and ranks.
     *  @param[in] Is array of modes.
     *  @param[in] Rs array of ranks.
     */
    TensorTrain(std::array<size_type, D> Is, std::array<size_type, D + 1> Rs)
            : modes_(Is)
            , ranks_(Rs) {
        for (auto K = 0; K < D; ++K) {
            shapes_[K] = {ranks_[K], modes_[K], ranks_[K + 1]};
            cores_[K] = core_type(shapes_[K]).setZero();
        }
    }

    /** Zero-initialize a tensor train from given shapes.
     *  @param[in] Ss array of shapes.
     */
    explicit TensorTrain(std::array<size_type, D> Ss)
            : shapes_(Ss) {
        for (auto K = 0; K < D; ++K) {
            ranks_[K] = shapes_[K][0];
            modes_[K] = shapes_[K][1];
            ranks_[K + 1] = shapes_[K][2];
            cores_[K] = core_type(shapes_[K]).setZero();
        }
    }

    /** *Destructively* generate a tensor train from given data, modes, and
     * tolerance.
     *  @param[in] A dense tensor data in *natural descending order*
     *  @param[in] Is array of modes.
     *  @param[in] epsilon decomposition tolerance.
     *
     *  Given a full tensor \f$\mathcal{X}\f$ and its TT decomposition
     * \f$\bar{\mathcal{X}}\f$, the latter is constructed such that:
     *
     *  \f[
     *     \| \mathcal{X} - \tilde{\mathcal{X}} \| \leq \epsilon \| \mathcal{X} \|
     *  \f]
     *
     *  where \f$\tilde{\mathcal{X}}\f$ is the reconstructed full tensor and
     *  \f$\| \cdot \|\f$ the Frobenius norm.
     *  \f$\epsilon\f$ is user-provided and we compute the SVD truncation threshold
     * from it, as follows:
     *
     *  \f[
     *     \delta = \frac{\epsilon}{\sqrt{D-1}} \| \mathcal{X} \|
     *  \f]
     *
     *  @warning The dense tensor data is assumed to be in natural descending
     *  order. This is critical for the tensor train SVD algorithm to work correctly.
     */
    TensorTrain(T *A, std::array<size_type, D> Is, T epsilon = 1e-12)
            : norm_computed_{true}
            , epsilon_{epsilon}
            , modes_{Is} {
        u_count_ = std::accumulate(
            modes_.cbegin(), modes_.cend(), std::multiplies<size_type>(), 1);
        // compute norm of input tensor
        norm_ = frobenius_norm(A, u_count_);

        auto delta = epsilon_ * norm_ / std::sqrt(D - 1);

        // the "border" ranks are 1 by construction
        ranks_.front() = ranks_.back() = 1;

        decompose(A, delta);
    }

    /** *Destructively* generate a tensor train from given tensor and tolerance.
     *  @param[in] A dense tensor data in *natural descending order*
     *  @param[in] epsilon decomposition tolerance.
     *
     *  Given a full tensor \f$\mathcal{X}\f$ and its TT decomposition
     * \f$\bar{\mathcal{X}}\f$, the latter is constructed such that:
     *
     *  \f[
     *     \| \mathcal{X} - \tilde{\mathcal{X}} \| \leq \epsilon \| \mathcal{X} \|
     *  \f]
     *
     *  where \f$\tilde{\mathcal{X}}\f$ is the reconstructed full tensor and
     *  \f$\| \cdot \|\f$ the Frobenius norm.
     *  \f$\epsilon\f$ is user-provided and we compute the SVD truncation threshold
     * from it, as follows:
     *
     *  \f[
     *     \delta = \frac{\epsilon}{\sqrt{D-1}} \| \mathcal{X} \|
     *  \f]
     *
     *  @warning The dense tensor data is assumed to be in natural descending
     *  order. This is critical for the tensor train SVD algorithm to work correctly.
     */
    TensorTrain(Eigen::Tensor<T, D> &A, T epsilon = 1e-12)
            : norm_computed_{true}
            , epsilon_{epsilon}
            , modes_{A.dimensions()} {
        u_count_ = A.size();
        // compute norm of input tensor
        norm_ = frobenius_norm(A.data(), u_count_);

        auto delta = epsilon_ * norm_ / std::sqrt(D - 1);

        // the "border" ranks are 1 by construction
        ranks_.front() = ranks_.back() = 1;

        decompose(A.data(), delta);
    }

    /** *Non-destructively* generate a tensor train from given tensor and tolerance.
     *  @param[in] A dense tensor data in *natural descending order*
     *  @param[in] epsilon decomposition tolerance.
     *
     *  Given a full tensor \f$\mathcal{X}\f$ and its TT decomposition
     * \f$\bar{\mathcal{X}}\f$, the latter is constructed such that:
     *
     *  \f[
     *     \| \mathcal{X} - \tilde{\mathcal{X}} \| \leq \epsilon \| \mathcal{X} \|
     *  \f]
     *
     *  where \f$\tilde{\mathcal{X}}\f$ is the reconstructed full tensor and
     *  \f$\| \cdot \|\f$ the Frobenius norm.
     *  \f$\epsilon\f$ is user-provided and we compute the SVD truncation threshold
     * from it, as follows:
     *
     *  \f[
     *     \delta = \frac{\epsilon}{\sqrt{D-1}} \| \mathcal{X} \|
     *  \f]
     *
     *  @warning The dense tensor data is assumed to be in natural descending
     *  order. This is critical for the tensor train SVD algorithm to work correctly.
     */
    TensorTrain(const Eigen::Tensor<T, D> &A, T epsilon = 1e-12)
            : norm_computed_{true}
            , epsilon_{epsilon}
            , modes_{A.dimensions()} {
        // take copy of input tensor, so the tensor train is generated
        // non-destructively.
        Eigen::Tensor<T, D> B = A;

        u_count_ = B.size();
        // compute norm of input tensor
        norm_ = frobenius_norm(B.data(), u_count_);

        auto delta = epsilon_ * norm_ / std::sqrt(D - 1);

        // the "border" ranks are 1 by construction
        ranks_.front() = ranks_.back() = 1;

        decompose(B.data(), delta);
    }
    /*\@}*/

    /*\@{ Tensor train operations
     *
     * @note It is not necessary to implement left orthonormalization.
     */
    /** *Destructive* right-orthonormalization of `*this` tensor train.
     *
     * This is an implementation of algorithm 2.1 in: Al Daas, H.; Ballard, G.;
     * Benner, P. Parallel Algorithms for Tensor Train Arithmetic. arXiv
     * [math.NA], 2020.
     *
     * @note We use the Householder QR algorithm, as implemented in Eigen.
     */
    void right_orthonormalize() {
        // start from last core and go down to second mode
        for (auto i = D - 1; i > 0; --i) {
            // shape of horizontal unfolding of current, i-th, core
            auto h_rows = shapes_[i][0];
            auto h_cols = shapes_[i][1] * shapes_[i][2];
            // *adjoint* of horizontal unfolding of current, i-th, core
            matrix_type Ht =
                Eigen::Map<matrix_type>(cores_[i].data(), h_rows, h_cols).adjoint();

            // Householder QR decomposition
            Eigen::HouseholderQR<matrix_type> qr(Ht);

            // sequence of Householder reflectors
            auto hh = qr.householderQ();
            // Q1 is the *thin* Q factor and Q1t its adjoint.
            matrix_type Q1t = matrix_type::Identity(h_rows, h_cols);
            // we compute it by applying hh on the right
            // of the correctly dimensioned identity matrix
            Q1t.applyOnTheRight(hh.adjoint());
            // set the result to be the *adjoint* of the horizontal unfolding of
            // current, i-th, core
            std::copy(Q1t.data(), Q1t.data() + cores_[i].size(), cores_[i].data());

            // R1 factor (*thin* R)
            matrix_type R1 = qr.matrixQR()
                                 .topLeftCorner(h_rows, h_rows)
                                 .template triangularView<Eigen::Upper>();

            // shape of vertical unfolding of next, (i-1)-th, core
            auto v_rows = shapes_[i - 1][0] * shapes_[i - 1][1];
            auto v_cols = shapes_[i - 1][2];
            // vertical unfolding of next, (i-1)-th, core times adjoint of R factor
            matrix_type next =
                Eigen::Map<matrix_type>(cores_[i - 1].data(), v_rows, v_cols) *
                R1.adjoint();
            // set the result to be the vertical unfolding of the next core
            std::copy(next.data(),
                      next.data() + cores_[i - 1].size(),
                      cores_[i - 1].data());
        }

        is_orthonormal_ = true;

        // compute norm while we're at it
        norm_ = frobenius_norm(cores_[0].data(), cores_[0].size());
        norm_computed_ = true;
    }
    /** Rounding of `*this` tensor train.
     *
     * @param[in] epsilon decomposition tolerance.
     *
     *  Given a tensor in TT format \f$\mathcal{Y}\f$, rounding produces a tensor in
     * TT format \f$\mathcal{X}\f$ with reduced ranks, sucht that:
     *
     *  \f[
     *     \| \tilde{\mathcal{X}} - \tilde{\mathcal{Y}} \| \leq \epsilon \|
     * \tilde{\mathcal{Y}} \| \f]
     *
     *  where \f$\tilde{\cdot}\f$ are the reconstructed full tensors and
     *  \f$\| \cdot \|\f$ the Frobenius norm.
     *  \f$\epsilon\f$ is user-provided and we compute the SVD truncation threshold
     * from it, as follows:
     *
     *  \f[
     *     \delta = \frac{\epsilon}{\sqrt{D-1}} \| \tilde{\mathcal{Y}} \|
     *  \f]
     *
     * This is an implementation of algorithm 2.2 in: Al Daas, H.; Ballard, G.;
     * Benner, P. Parallel Algorithms for Tensor Train Arithmetic. arXiv
     * [math.NA], 2020. The description of algorithm first appeared as algorithm 2
     * in: Oseledets, I. V. Tensor-Train Decomposition. SIAM J. Sci. Comput. 2011, 33
     * (5), 2295–2317. https://doi.org/10.1137/090752286.
     *
     * @note We use the block divide-and-conquer SVD algorithm, as implemented in
     * Eigen.
     */
    void round(T epsilon) {
        // reset epsilon_
        epsilon_ = epsilon;

        // Check whether we have the norm of *this already, because:
        // a. either we are rounding right after decomposing (bit pointless, but...),
        // b. or *this is already right-orthonormalized
        // if not, right-orhtonormalize and compute the norm.
        if (!norm_computed_ || !is_orthonormal_) { right_orthonormalize(); }
        auto delta = epsilon_ * norm_ / std::sqrt(D - 1);
        SPDLOG_INFO("SVD threshold = {:6e}", delta);

        ranks_.front() = ranks_.back() = 1;
        for (auto i = 0; i < D - 1; ++i) {
            // shape of vertical unfolding
            auto v_rows = ranks_[i] * modes_[i];
            auto v_cols = ranks_[i + 1];
            // vertical unfolding
            matrix_type V =
                Eigen::Map<matrix_type>(cores_[i].data(), v_rows, v_cols);

            // compute SVD of vertical unfolding
            auto start = Clock::now();
            auto svd = V.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
            auto stop = Clock::now();
            std::chrono::duration<double, std::milli> elapsed = stop - start;

            SPDLOG_INFO("SVD decomposition of mode {} in {}", i, elapsed);

            if (svd.info() != Eigen::Success) {
                SPDLOG_ERROR(
                    "SVD decomposition of mode {} (out of {}) did not succeed!",
                    i,
                    D);
                std::abort();
            }

            // re-define ranks and cores
            auto rank = (svd.singularValues().array() >= delta).count();
            ranks_[i + 1] = rank;
            shapes_[i] = {ranks_[i], modes_[i], ranks_[i + 1]};
            // resize current core
            cores_[i] = core_type(shapes_[i]);
            // only take the first rank columns of U to fill cores_[i]
            std::copy(svd.matrixU().data(),
                      svd.matrixU().data() + cores_[i].size(),
                      cores_[i].data());

            // shape of horizontal unfolding
            auto h_rows = shapes_[i + 1][0];
            auto h_cols = shapes_[i + 1][1] * shapes_[i + 1][2];
            // horizontal unfolding
            matrix_type H =
                Eigen::Map<matrix_type>(cores_[i + 1].data(), h_rows, h_cols);

            matrix_type next =
                svd.singularValues().head(rank).asDiagonal() *
                svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).adjoint() * H;

            // reshape next core
            cores_[i + 1] =
                core_type(shape_type{ranks_[i + 1],
                                     modes_[i + 1],
                                     H.size() / (ranks_[i + 1], modes_[i + 1])});
            // copy into next core
            std::copy(next.data(), next.data() + next.size(), cores_[i + 1].data());
        }
        // fix shape of last core
        shapes_[D - 1] = {ranks_[D - 1], modes_[D - 1], ranks_[D]};

        // reset compressed count of elements
        c_count_ = 0;
        for (const auto &c : cores_) { c_count_ += c.size(); }

        // after rounding, the tensor is not right-orthonormal anymore
        is_orthonormal_ = false;
    }
    /*\@}*/

    /*\@{ Arithmetic */
    /** *Non-destructive* scaling by a scalar
     *
     * @tparam U scalar type of the scalar.
     * @param[in] alpha scalar.
     * @return Y scaled tensor train.
     */
    template <typename U>
    auto scale(U alpha) const
        -> TensorTrain<typename std::common_type<U, T>::type, D> {
        static_assert(std::is_floating_point_v<U>,
                      "Scaling factor alpha can only be a floating point type!");

        // common type between U and T
        using V = typename std::common_type<U, T>::type;

        TensorTrain<V, D> Y = *this;
        // Eigen::Tensor does not implement *=
        Y.cores_[0] = alpha * Y.cores_[0];

        return Y;
    }

    /** *Destructive* scaling by a scalar
     *
     * @tparam U scalar type of the scalar.
     * @param[in] alpha scalar.
     */
    template <typename U> auto scale(U alpha) -> void {
        static_assert(std::is_floating_point_v<U>,
                      "Scaling factor alpha can only be a floating point type!");

        // Eigen::Tensor does not implement *=
        cores_[0] = alpha * cores_[0];
    }
    /*\@}*/

    /*\@{ Iterators */
    /*\@}*/

    /*\@{ Getters/setters */
    /** Get Frobenius norm of tensor.
     *
     * @note This function in non-`const` as we might need to right-orthonormalize to
     * compute the norm!
     */
    auto norm() -> T {
        if (!norm_computed_ || !is_orthonormal_) { right_orthonormalize(); }
        return norm_;
    }

    /** Get array of mode sizes. */
    auto modes() const -> std::array<size_type, D> { return modes_; }
    /** Get size of i-th mode.
     *
     * @param[in] i requested mode.
     */
    auto mode(std::size_t i) const -> size_type { return modes_[i]; }

    /** Get array of ranks. */
    auto ranks() const -> std::array<size_type, D + 1> { return ranks_; }
    /** Get i-th rank.
     *
     * @param[in] i requested mode.
     */
    auto rank(std::size_t i) const -> size_type { return ranks_[i]; }

    /** Get array of shapes of the tensor train cores. */
    auto shapes() const -> std::array<shape_type, D> { return shapes_; }
    /** Get shape of i-th tensor train core.
     *
     * @param[in] i requested core.
     */
    auto shape(std::size_t i) const -> shape_type { return shapes_[i]; }

    /** Get array of tensor train cores. */
    auto cores() const -> std::array<core_type, D> { return cores_; }
    /** Get mutable reference to i-th tensor train core.
     *
     * @param[in] i requested core.
     */
    auto core(std::size_t i) -> core_type & { return cores_[i]; }
    /** Get immutable reference to i-th tensor train core.
     *
     * @param[in] i requested core.
     */
    auto core(std::size_t i) const -> const core_type & { return cores_[i]; }

    /** Get maximum rank of the tensor train decomposition. */
    auto max_rank() const -> size_type { return std::max(ranks_); }
    /*\@}*/

    /*\@{ Size and count statistics */
    /** Get number of elements of compressed (tensor train) representation. */
    auto size() const -> double { return c_count_; }

    /** Get size, in GiB, of compressed (tensor train) representation. */
    auto GiB() const -> double { return to_GiB<T>(c_count_); }

    /** Get number of elements of uncompressed representation. */
    auto uncompressed_size() const -> size_type { return u_count_; }

    /** Get size, in GiB, of uncompressed representation. */
    auto uncompressed_GiB() const -> double { return to_GiB<T>(u_count_); }

    /** Get compression rate. */
    auto compression() const -> double {
        return (1.0 - static_cast<double>(c_count_) / static_cast<double>(u_count_));
    }
    /*\@}*/
    /** Reconstruct full tensor from `*this` tensor train. */
    auto to_full() const -> Eigen::Tensor<T, D> {
        Eigen::Tensor<T, D> full(modes_);

        extent_type<1> cdims = {index_pair_type(1, 0)};
        // contract dimension 1 of first chipped core with dimension 0 of second core
        Eigen::Tensor<T, 3> post = (cores_[0].chip(0, 0)).contract(cores_[1], cdims);

        // recursion
        to_full<2>(post, full);

        return full;
    }
};

/** *Non-destructive* scaling by a scalar (on the left)
 *
 * @tparam T scalar type of the tensor train.
 * @tparam U scalar type of the scalar.
 * @tparam D number of modes in the tensor train.
 * @param[in] alpha scalar.
 * @param[in] X tensor train.
 * @return Y scaled tensor train.
 */
template <typename T, typename U, int D>
auto operator*(U alpha, const TensorTrain<T, D> &X)
    -> TensorTrain<decltype(U() * T()), D> {
    return X.scale(alpha);
}

/** *Non-destructive* scaling by a scalar (on the right)
 *
 * @tparam T scalar type of the tensor train.
 * @tparam U scalar type of the scalar.
 * @tparam D number of modes in the tensor train.
 * @param[in] X tensor train.
 * @param[in] alpha scalar.
 * @return Y scaled tensor train.
 */
template <typename T, typename U, int D>
auto operator*(const TensorTrain<T, D> &X, U alpha)
    -> TensorTrain<decltype(T() * U()), D> {
    return X.scale(alpha);
}

/** Sum of two tensor trains, without rounding.
 *
 * @tparam T scalar type of the left tensor train.
 * @tparam U scalar type of the right tensor train.
 * @tparam D number of modes in the tensor trains.
 * @tparam V scalar type of the result tensor train.
 * @param[in] X tensor train.
 * @param[in] Y tensor train.
 * @return Z the sum tensor train.
 */
template <typename T,
          typename U,
          int D,
          typename V = typename std::common_type<U, T>::type>
auto sum(const TensorTrain<T, D> &X, const TensorTrain<U, D> &Y)
    -> TensorTrain<V, D> {
    using size_type = typename TensorTrain<V, D>::size_type;

    // left and right operands must be congruent: their modes array must be the same
    if (X.modes() != Y.modes()) { std::abort(); }

    // build up the shapes of the tensor cores for the sum
    // the ranks are the sum of those of X and Y, except for first and last
    std::array<size_type, D + 1> ranks;
    ranks.front() = ranks.back() = 1;
    for (auto i = 1; i < D; ++i) { ranks[i] = X.rank(i) + Y.rank(i); }

    TensorTrain<V, D> Z(X.modes(), ranks);

    Eigen::array<size_type, 3> off_X = {0, 0, 0};
    Eigen::array<size_type, 3> off_Y;
    // stack cores
    for (auto i = 0; i < D; ++i) {
        // left operand in "upper left" corner
        Z.core(i).slice(off_X, X.shape(i)) = X.core(i);

        // right operand in "lower right" corner
        off_Y = {Z.shape(i)[0] - Y.shape(i)[0], 0, Z.shape(i)[2] - Y.shape(i)[2]};
        Z.core(i).slice(off_Y, Y.shape(i)) = Y.core(i);
    }

    return Z;
}

/** Sum of two tensor trains, with rounding.
 *
 * @tparam T scalar type of the left tensor train.
 * @tparam U scalar type of the right tensor train.
 * @tparam D number of modes in the tensor trains.
 * @tparam V scalar type of the result tensor train.
 * @param[in] X tensor train.
 * @param[in] Y tensor train.
 * @param[in] epsilon rounding threshold.
 * @return Z the sum tensor train.
 */
template <typename T,
          typename U,
          int D,
          typename V = typename std::common_type<U, T>::type>
auto sum(const TensorTrain<T, D> &X, const TensorTrain<U, D> &Y, V epsilon)
    -> TensorTrain<V, D> {
    auto Z = sum(X, Y);

    // perform rounding.
    Z.round(epsilon);

    return Z;
}

/** Hadamard (elementwise) product of two tensor trains, without rounding.
 *
 * @tparam T scalar type of the left tensor train.
 * @tparam U scalar type of the right tensor train.
 * @tparam D number of modes in the tensor trains.
 * @tparam V scalar type of the result tensor train.
 * @param[in] X tensor train.
 * @param[in] Y tensor train.
 * @return Z the elementwise product tensor train.
 */
template <typename T,
          typename U,
          int D,
          typename V = typename std::common_type<U, T>::type>
auto hadamard_product(const TensorTrain<T, D> &X, const TensorTrain<U, D> &Y)
    -> TensorTrain<V, D> {
    using size_type = typename TensorTrain<V, D>::size_type;

    // left and right operands must be congruent: their modes array must be the same
    if (X.modes() != Y.modes()) { std::abort(); }

    // build up the shapes of the tensor cores for the sum
    // the ranks are the sum of those of X and Y, except for first and last
    std::array<size_type, D + 1> ranks;
    for (auto i = 0; i < D + 1; ++i) { ranks[i] = X.rank(i) * Y.rank(i); }

    TensorTrain<V, D> Z(X.modes(), ranks);

    // compute cores as the tensor product of the slices

    using index_pair_type = typename TensorTrain<V, D>::index_pair_type;
    using extent_type = typename TensorTrain<V, D>::template extent_type<1>;

    extent_type cdims = {index_pair_type(1, 1)};

    //  offset of slices
    Eigen::array<size_type, 3> offs;
    // extents of slices
    Eigen::array<size_type, 3> ext_X, ext_Y, ext_Z;
    // shuffle
    Eigen::array<size_type, 4> shuffle = {0, 2, 1, 3};

    // loop over cores
    for (auto i = 0; i < D; ++i) {
        // loop over slices
        for (auto j = 0; j < Z.mode(i); ++j) {
            // offset of operands and result slices
            offs = {0, j, 0};
            // extent of slice for left operand
            ext_X = {X.rank(i), 1, X.rank(i + 1)};
            // extent of slice for right operand
            ext_Y = {Y.rank(i), 1, Y.rank(i + 1)};
            // extent of slice for result
            ext_Z = {Z.rank(i), 1, Z.rank(i + 1)};

            Z.core(i).slice(offs, ext_Z) =
                X.core(i)
                    .slice(offs, ext_X)
                    .contract(Y.core(i).slice(offs, ext_Y), cdims)
                    .shuffle(shuffle)
                    .reshape(ext_Z);
        }
    }

    return Z;
}

/** Hadamard (elementwise) product of two tensor trains, with rounding.
 *
 * @tparam T scalar type of the left tensor train.
 * @tparam U scalar type of the right tensor train.
 * @tparam D number of modes in the tensor trains.
 * @tparam V scalar type of the result tensor train.
 * @param[in] X tensor train.
 * @param[in] Y tensor train.
 * @param[in] epsilon rounding threshold.
 * @return Z the elementwise product tensor train.
 */
template <typename T,
          typename U,
          int D,
          typename V = typename std::common_type<U, T>::type>
auto hadamard_product(const TensorTrain<T, D> &X,
                      const TensorTrain<U, D> &Y,
                      V epsilon) -> TensorTrain<V, D> {
    auto Z = hadamard_product(X, Y);

    // perform rounding.
    Z.round(epsilon);

    return Z;
}
} // namespace tteigen
