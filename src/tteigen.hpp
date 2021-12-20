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

template <typename T, int D> class TT final {
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
        matrix_type next =
            svd.singularValues().head(rank).asDiagonal() *
            svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).transpose();

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
                   svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).transpose();
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
    TT() = default;

    /** Zero-initialize a tensor train from given modes and ranks.
     *  @param[in] Is array of modes.
     *  @param[in] Rs array of ranks.
     */
    TT(std::array<size_type, D> Is, std::array<size_type, D + 1> Rs)
            : modes_(Is)
            , ranks_(Rs) {
        for (auto K = 0; K < D; ++K) {
            shapes_[K] = {ranks_[K], modes_[K], ranks_[K + 1]};
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
    TT(T *A, std::array<size_type, D> Is, T epsilon = 1e-12)
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
    TT(Eigen::Tensor<T, D> &A, T epsilon = 1e-12)
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
    TT(const Eigen::Tensor<T, D> &A, T epsilon = 1e-12)
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
     * @note We use the Householder QR algorithm, as implemented in Eigen. The
     * successive QR decompositions are done in-place: the original data is
     * destroyed in the right-orthonormalization process.
     */
    void right_orthonormalize() {
        // start from last core and go down to second mode
        for (auto i = D - 1; i > 0; --i) {
            // shape of horizontal unfolding of current, i-th, core
            auto h_rows = shapes_[i][0];
            auto h_cols = shapes_[i][1] * shapes_[i][2];
            // *transpose* of horizontal unfolding of current, i-th, core
            matrix_type Ht =
                Eigen::Map<matrix_type>(cores_[i].data(), h_rows, h_cols)
                    .transpose();

            // in-place Householder QR decomposition
            Eigen::HouseholderQR<Eigen::Ref<matrix_type>> qr(Ht);

            // sequence of Householder reflectors
            auto hh = qr.householderQ();
            // Q1 is the *thin* Q factor and Q1t its transpose.
            matrix_type Q1t = matrix_type::Identity(h_rows, h_cols);
            // we compute it by applying hh on the right
            // of the correctly dimensioned identity matrix
            Q1t.applyOnTheRight(hh.transpose());
            // set the result to be the *transpose* of the horizontal unfolding of
            // current, i-th, core
            std::copy(Q1t.data(), Q1t.data() + cores_[i].size(), cores_[i].data());

            // R1 factor (*thin* R)
            matrix_type R1 = qr.matrixQR()
                                 .topLeftCorner(h_rows, h_rows)
                                 .template triangularView<Eigen::Upper>();

            // shape of vertical unfolding of next, (i-1)-th, core
            auto v_rows = shapes_[i - 1][0] * shapes_[i - 1][1];
            auto v_cols = shapes_[i - 1][2];
            // vertical unfolding of next, (i-1)-th, core times transpose of R factor
            matrix_type next =
                Eigen::Map<matrix_type>(cores_[i - 1].data(), v_rows, v_cols) *
                R1.transpose();
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
    void rounding(T epsilon) {
        // reset epsilon_
        epsilon_ = epsilon;
        // Check whether we have the norm of *this already, because:
        // a. either we are rounding right after decomposing (bit pointless, but...),
        // b. or *this is already right-orthonormalized
        if (!norm_computed_ || !is_orthonormal_) { right_orthonormalize(); }
        auto delta = epsilon_ * norm_ / std::sqrt(D - 1);
        // FIXME
    }
    /*\@}*/

    /*\@{ Arithmetic */
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

    /*\@{ Unfoldings */
    /** Horizontal unfolding of i-th tensor core.
     *
     * @param[in] i core to unfold
     *
     *  Given a 3-mode tensor \f$\mathcal{T} \in \mathbb{K}^{N\times L \times
     *  M}\f$, the horizontal unfolding generates a matrix
     *  \f$\mathcal{H}(\mathcal{T}) \in \mathbb{K}^{N\times LM}$ by
     *  concatenating the slices \f$\mathbf{X}_{\mathcal{T}}(:, l, :) \in
     *  \mathbb{K}^{N\times M}\f$ _horizontally_.
     */
    auto horizontal_unfolding(std::size_t i) const -> matrix_type {
        const auto n_rows = shapes_[i][0];
        const auto n_cols = shapes_[i][1] * shapes_[i][2];

        return Eigen::Map<matrix_type>(cores_[i].data(), n_rows, n_cols);
    }

    /** Vertical unfolding of i-th tensor core.
     *
     *  Given a 3-mode tensor \f$\mathcal{T} \in \mathbb{K}^{N\times L \times M}\f$,
     *  the vertical unfolding generates a matrix \f$\mathcal{V}(\mathcal{T}) \in
     * \mathbb{K}^{NL\times M}$ by concatenating the slices
     * \f$\mathbf{X}_{\mathcal{T}}(:, l, :) \in \mathbb{K}^{N\times M}\f$
     * _vertically_.
     */
    auto vertical_unfolding(std::size_t i) const -> matrix_type {
        const auto n_rows = shapes_[i][0] * shapes_[i][1];
        const auto n_cols = shapes_[i][2];

        return Eigen::Map<matrix_type>(cores_[i].data(), n_rows, n_cols);
    }
    /*\@}*/
};

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
    const double A_F = frobenius_norm(A.data(), A.size());

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
