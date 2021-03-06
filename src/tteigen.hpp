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

    /** Norm of the tensor train. */
    T norm_{0};

    /** Decomposition threshold. */
    double epsilon_{1e-12};

    /** Sizes of tensor modes \f$I_{n}\f$ */
    std::array<size_type, D> modes_;

    /** Tensor train cores: \f$\mathcal{T}_{\mathcal{X}, n} \in \mathbb{K}^{R_{n-1}
     * \times I_{n} \times R_{n}}\f$ */
    std::array<core_type, D> cores_;

    /** Number of rows and columns of horizontal unfolding of given core.
     *
     * @param[in] i index of the core
     */
    std::tuple<size_type, size_type> horizontal(std::size_t i) {
        return std::make_tuple(cores_[i].dimension(0),
                               cores_[i].dimension(1) * cores_[i].dimension(2));
    }

    /** Number of rows and columns of vertical unfolding of given core.
     *
     * @param[in] i index of the core
     */
    std::tuple<size_type, size_type> vertical(std::size_t i) {
        return std::make_tuple(cores_[i].dimension(0) * cores_[i].dimension(1),
                               cores_[i].dimension(2));
    }

    /** Tensor train decomposition *via* successive SVDs
     *
     *  @param[in] A dense tensor data in *natural descending order*.
     *  @param[in] sz size of dense tensor A.
     *  @param[in] delta SVD truncation threshold.
     *
     * This is an implementation of algorithm 1 in: Oseledets, I. V. Tensor-Train
     * Decomposition. SIAM J. Sci. Comput. 2011, 33 (5), 2295???2317.
     * https://doi.org/10.1137/090752286.
     *
     * @note We use the block divide-and-conquer SVD algorithm, as implemented in
     * Eigen.
     */
    void decompose(T *A, size_type sz, T delta) {
        using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        SPDLOG_INFO("SVD threshold = {:6e}", delta);

        // prepare first horizontal unfolding
        auto n_rows = modes_[0];
        auto n_cols = sz / n_rows;

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

        // define rank
        auto rank = (svd.singularValues().array() >= delta).count();
        // only take the first rank columns of U to fill cores_[0]
        matrix_type U = svd.matrixU()(Eigen::all, Eigen::seqN(0, rank));
        cores_[0] =
            Eigen::TensorMap<core_type>(U.data(), shape_type{1, modes_[0], rank});

        // prepare next unfolding: only use first rank singular values and first rank
        // columns of V
        matrix_type SigmaVh =
            svd.singularValues().head(rank).asDiagonal() *
            svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).adjoint();

        // go through the modes (dimensions) in the tensor
        for (int K = 1; K < D - 1; ++K) {
            // sizes of tensor unfoldings
            n_rows = modes_[K];
            n_cols /= n_rows;
            // construct unfolding (into memory being held by M)
            new (&M) Eigen::Map<matrix_type>(SigmaVh.data(), rank * n_rows, n_cols);
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

            // update rank and cores
            rank = (svd.singularValues().array() >= delta).count();
            // only take the first rank_R columns of U to fill cores_[K]
            U = svd.matrixU()(Eigen::all, Eigen::seqN(0, rank));
            cores_[K] = Eigen::TensorMap<core_type>(
                U.data(),
                shape_type{U.size() / (modes_[K] * rank), modes_[K], rank});

            // prepare next unfolding: only use first rank singular values and first
            // rank columns of V
            SigmaVh = svd.singularValues().head(rank).asDiagonal() *
                      svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).adjoint();
        }

        // fill cores_[D-1]
        start = Clock::now();
        cores_[D - 1] = Eigen::TensorMap<core_type>(
            SigmaVh.data(), shape_type{rank, modes_[D - 1], 1});
        stop = Clock::now();
        elapsed = stop - start;
        SPDLOG_INFO("SVD decomposition of mode {} in {}", D - 1, elapsed);
    }

    /** Tensor train decomposition *via* successive randomized SVDs.
     *
     *  @param[in] A dense tensor data in *natural descending order*.
     *  @param[in] sz size of dense tensor A.
     *
     * This is an implementation of algorithm 1 in: Oseledets, I. V. Tensor-Train
     * Decomposition. SIAM J. Sci. Comput. 2011, 33 (5), 2295???2317.
     * https://doi.org/10.1137/090752286.
     *
     * @note We use the randomized SVD algorithm:
     * Halko, N.; Martinsson, P. G.; Tropp, J. A. Finding Structure with Randomness:
     * Probabilistic Algorithms for Constructing Approximate Matrix Decompositions.
     * SIAM Rev. 2011, 53 (2), 217???288. https://doi.org/10.1137/090771806.
     */
    void decompose(T *A, size_type sz) {
        using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        // prepare first horizontal unfolding
        auto n_rows = modes_[0];
        auto rank = 2;
        auto n_cols = sz / n_rows;

        // wrap tensor data into a matrix
        Eigen::Map<matrix_type> M(A, n_rows, n_cols);

        // compute SVD of unfolding
        auto start = Clock::now();
        auto [U, Sigma, V] = randomized_svd(M, rank);
        auto stop = Clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;

        SPDLOG_INFO("Randomized SVD decomposition of mode {} in {}", 0, elapsed);

        // update cores
        cores_[0] =
            Eigen::TensorMap<core_type>(U.data(), shape_type{1, modes_[0], rank});

        // prepare next unfolding: only use first rank singular values and first rank
        // columns of V
        matrix_type SigmaVh = Sigma * V.adjoint();

        // go through the modes (dimensions) in the tensor
        for (int K = 1; K < D - 1; ++K) {
            // sizes of tensor unfoldings
            n_rows = modes_[K];
            n_cols /= n_rows;
            // construct unfolding (into memory being held by M)
            new (&M) Eigen::Map<matrix_type>(SigmaVh.data(), rank * n_rows, n_cols);
            // compute SVD of unfolding
            // rank = modes_[K] / 2;
            start = Clock::now();
            auto [U, Sigma, V] = randomized_svd(M, rank);
            stop = Clock::now();
            elapsed = stop - start;
            SPDLOG_INFO("Randomized SVD decomposition of mode {} in {}", K, elapsed);

            // update cores
            cores_[K] = Eigen::TensorMap<core_type>(
                U.data(),
                shape_type{U.size() / (modes_[K] * rank), modes_[K], rank});

            // prepare next unfolding: only use first rank singular values and first
            // rank columns of V
            SigmaVh = Sigma * V.adjoint();
        }

        // fill cores_[D-1]
        start = Clock::now();
        cores_[D - 1] = Eigen::TensorMap<core_type>(
            SigmaVh.data(), shape_type{rank, modes_[D - 1], 1});
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
        extent_type<1> c_axes = {index_pair_type(mode, 0)};

        Eigen::Tensor<T, mode + 2> post = pre.contract(cores_[mode], c_axes);

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
        extent_type<1> c_axes = {index_pair_type(D - 1, 0)};

        // the contraction would be an Eigen::Tensor<T, D+1>, with last mode of size
        // 1, hence the chipping.
        full = pre.contract(cores_[D - 1], c_axes).chip(0, D);
    }
    /*\@}*/

public:
    /*\@{ Constructors */
    TensorTrain() = default;

    /** Zero-initialize a tensor train from given modes and ranks.
     *  @param[in] Is array of modes.
     *  @param[in] Rs array of ranks.
     */
    TensorTrain(const std::array<size_type, D> &Is,
                const std::array<size_type, D + 1> &Rs)
            : modes_(Is) {
        for (auto K = 0; K < D; ++K) {
            cores_[K] = core_type(shape_type{Rs[K], modes_[K], Rs[K + 1]}).setZero();
        }
    }

    /** Zero-initialize a tensor train from given shapes.
     *  @param[in] Ss array of shapes.
     */
    explicit TensorTrain(const std::array<shape_type, D> &Ss) {
        for (auto K = 0; K < D; ++K) {
            modes_[K] = Ss[K][1];
            cores_[K] = core_type(Ss[K]).setZero();
        }
    }

    /** Initialize a tensor train from given cores.
     *  @param[in] Cs array of cores.
     *  @note This is mainly useful for testing.
     */
    explicit TensorTrain(const std::array<core_type, D> &Cs)
            : cores_(Cs) {
        for (auto K = 0; K < D; ++K) { modes_[K] = cores_[K].dimension(1); }
    }

    /** *Destructively* generate a tensor train from given data, modes, and
     * tolerance.
     *
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
    TensorTrain(T *A, std::array<size_type, D> Is, double epsilon)
            : norm_computed_{true}
            , epsilon_{epsilon}
            , modes_{Is} {
        auto sz = std::accumulate(
            modes_.cbegin(), modes_.cend(), 1, std::multiplies<size_type>());
        // compute norm of input tensor
        norm_ = frobenius_norm(A, sz);

        auto delta = epsilon_ * norm_ / std::sqrt(D - 1);

        decompose(A, sz, delta);
    }

    /** *Destructively* generate a tensor train from given data and modes, through
     * successive randomized SVDs.
     *
     *  @param[in] A dense tensor data in *natural descending order*
     *  @param[in] Is array of modes.
     *
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
     *
     *  @warning The dense tensor data is assumed to be in natural descending
     *  order. This is critical for the tensor train SVD algorithm to work correctly.
     */
    TensorTrain(T *A, std::array<size_type, D> Is)
            : norm_computed_{true}
            , modes_{Is} {
        auto sz = std::accumulate(
            modes_.cbegin(), modes_.cend(), 1, std::multiplies<size_type>());
        // compute norm of input tensor
        norm_ = frobenius_norm(A, sz);

        decompose(A, sz);
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
    TensorTrain(Eigen::Tensor<T, D> &A, double epsilon = 1e-12)
            : norm_computed_{true}
            , epsilon_{epsilon}
            , modes_{A.dimensions()} {
        // compute norm of input tensor
        norm_ = frobenius_norm(A.data(), A.size());

        auto delta = epsilon_ * norm_ / std::sqrt(D - 1);

        decompose(A.data(), A.size(), delta);
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
    TensorTrain(const Eigen::Tensor<T, D> &A, double epsilon)
            : norm_computed_{true}
            , epsilon_{epsilon}
            , modes_{A.dimensions()} {
        // take copy of input tensor, so the tensor train is generated
        // non-destructively.
        Eigen::Tensor<T, D> B = A;

        // compute norm of input tensor
        norm_ = frobenius_norm(B.data(), B.size());

        auto delta = epsilon_ * norm_ / std::sqrt(D - 1);

        decompose(B.data(), B.size(), delta);
    }

    /** *Non-destructively* generate a tensor train from given tensor, through
     * successive randomized SVDs.
     *
     *  @param[in] A dense tensor data in *natural descending order*
     *  @param[in] epsilon decomposition tolerance.
     *
     *  @warning The dense tensor data is assumed to be in natural descending
     *  order. This is critical for the tensor train SVD algorithm to work correctly.
     */
    TensorTrain(const Eigen::Tensor<T, D> &A)
            : norm_computed_{true}
            , modes_{A.dimensions()} {
        // take copy of input tensor, so the tensor train is generated
        // non-destructively.
        Eigen::Tensor<T, D> B = A;

        // compute norm of input tensor
        norm_ = frobenius_norm(B.data(), B.size());

        decompose(B.data(), B.size());
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
     * @note We use the Householder QR algorithm, as implemented in Eigen. If, for
     * core \f$n\f$, \f$R_{n}\f$ or \f$R_{n+1}\f$ are larger than \f$I_{n}\f$, this
     * operation does compress the representation.
     * This might happen when forming sums or Hadamard products of tensors.
     */
    void orthogonalize_RL() {
        // start from last core and go down to second mode
        for (auto i = D - 1; i > 0; --i) {
            // shape of horizontal unfolding of current, i-th, core
            auto [h_rows, h_cols] = horizontal(i);
            // whether to do thin or full QR
            bool do_thin_qr = (h_cols >= h_rows);
            // *adjoint* of horizontal unfolding of current, i-th, core
            matrix_type Ht =
                Eigen::Map<matrix_type>(cores_[i].data(), h_rows, h_cols).adjoint();

            // Householder QR decomposition, in place
            Eigen::HouseholderQR<Eigen::Ref<matrix_type>> qr(Ht);

            // sequence of Householder reflectors
            auto hh = qr.householderQ();
            // Qh is the adjoint of the Q factor (orthogonal)
            matrix_type Qh;
            if (do_thin_qr) {
                Qh = matrix_type::Identity(h_rows, h_cols);
            } else {
                Qh = matrix_type::Identity(h_cols, h_cols);
            }
            // we compute it by applying hh on the right
            // of the correctly dimensioned identity matrix
            Qh.applyOnTheRight(hh.adjoint());
            // set the result to be the *adjoint* of the horizontal unfolding of
            // current, i-th, core
            cores_[i] = Eigen::TensorMap<core_type>(
                Qh.data(), shape_type{Qh.rows(), modes_[i], Qh.cols() / modes_[i]});

            // R factor (upper triangular)
            matrix_type R;
            if (do_thin_qr) {
                R = qr.matrixQR()
                        .topLeftCorner(h_rows, h_rows)
                        .template triangularView<Eigen::Upper>();
            } else {
                R = qr.matrixQR().template triangularView<Eigen::Upper>();
            }

            // the next core is the mode-2 product of itself with the adjoint of R
            // by the definition of the mode-2 product, we can use R itself :)
            extent_type<1> c_axes = {index_pair_type(2, 1)};
            core_type next = cores_[i - 1].contract(
                Eigen::TensorMap<Eigen::Tensor<T, 2>>(R.data(), R.rows(), R.cols()),
                c_axes);
            SPDLOG_INFO("next.dimensions() {}", next.dimensions());
            // it seems it can't be done as: cores_[i-1] = cores_[i-1].contract(..)
            cores_[i - 1] = next;
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
     * (5), 2295???2317. https://doi.org/10.1137/090752286.
     *
     * @note We use the block divide-and-conquer SVD algorithm, as implemented in
     * Eigen.
     */
    void round(double epsilon) {
        // reset epsilon_
        epsilon_ = epsilon;

        // Check whether we have the norm of *this already, because:
        // a. either we are rounding right after decomposing (bit pointless, but...),
        // b. or *this is already right-orthonormalized
        // if not, right-orhtonormalize and compute the norm.
        if (!norm_computed_ || !is_orthonormal_) { orthogonalize_RL(); }
        auto delta = epsilon_ * norm_ / std::sqrt(D - 1);
        SPDLOG_INFO("SVD threshold = {:6e}", delta);

        for (auto i = 0; i < D - 1; ++i) {
            // shape of vertical unfolding
            auto [v_rows, v_cols] = vertical(i);
            //  vertical unfolding
            Eigen::Map<matrix_type> V(cores_[i].data(), v_rows, v_cols);

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

            // define rank and cores
            auto rank = (svd.singularValues().array() >= delta).count();
            // only take the first rank columns of U to fill cores_[i]
            matrix_type U = svd.matrixU()(Eigen::all, Eigen::seqN(0, rank));
            cores_[i] = Eigen::TensorMap<core_type>(
                U.data(),
                shape_type{U.size() / (modes_[i] * rank), modes_[i], rank});

            matrix_type SigmaVh =
                svd.singularValues().head(rank).asDiagonal() *
                svd.matrixV()(Eigen::all, Eigen::seqN(0, rank)).adjoint();

            // form the mode-0 product between the next core and the \f$\Sigma
            // V^{\dagger}\f$ matrix we need a shuffle to make the result of the
            // correct shape
            extent_type<1> c_axes = {index_pair_type(0, 1)};
            Eigen::array<size_type, 3> shuffle = {2, 0, 1};
            core_type next =
                cores_[i + 1]
                    .contract(Eigen::TensorMap<Eigen::Tensor<T, 2>>(
                                  SigmaVh.data(), SigmaVh.rows(), SigmaVh.cols()),
                              c_axes)
                    .shuffle(shuffle);
            // it seems it can't be done as: cores_[i+1] = cores_[i+1].contract(..)
            cores_[i + 1] = next;
        }

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
        if (!norm_computed_ || !is_orthonormal_) { orthogonalize_RL(); }
        return norm_;
    }

    /** Get array of mode sizes. */
    auto modes() const -> std::array<size_type, D> { return modes_; }
    /** Get size of i-th mode.
     *
     * @param[in] i requested mode.
     */
    auto mode(std::size_t i) const -> size_type { return modes_[i]; }

    /** Get array of ranks.
     *
     * This is a (D+1)-length array of the ranks \f$R_{n}\f$,
     * with \f$R_{0} = 1 = R_{n}\f$.
     */
    auto ranks() const -> std::array<size_type, D + 1> {
        std::array<size_type, D + 1> ranks;
        ranks.front() = ranks.back() = 1;
        for (auto i = 0; i < cores_.size() - 1; ++i) {
            ranks[i + 1] = cores_[i].dimension(2);
        }
        return ranks;
    }
    /** Get i-th rank.
     *
     * @param[in] i requested mode.
     */
    auto rank(std::size_t i) const -> size_type { return ranks()[i]; }

    /** Get array of shapes of the tensor train cores.
     *
     * Each shape is a 3-membered array \f$\lbrace R_{n-1}, I_{n}, R_{n} \rbrace\f$,
     * with \f$R_{0} = 1 = R_{n}\f$.
     */
    auto shapes() const -> std::array<shape_type, D> {
        std::array<shape_type, D> shapes;
        for (auto i = 0; i < D; ++i) { shapes[i] = cores_[i].dimensions(); }
        return shapes;
    }
    /** Get shape of i-th tensor train core.
     *
     * @param[in] i requested core.
     *
     * Shape of tensor core i \f$\lbrace R_{i-1}, I_{i}, R_{i} \rbrace\f$
     */
    auto shape(std::size_t i) const -> shape_type { return shapes()[i]; }

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
    auto max_rank() const -> size_type { return std::max(ranks()); }
    /*\@}*/

    /*\@{ Size and count statistics */
    /** Get number of elements of compressed (tensor train) representation. */
    auto size() const -> size_type {
        return std::accumulate(
            cores_.cbegin(), cores_.cend(), 0, [](size_type sz, const core_type &Y) {
                return sz + Y.size();
            });
    }

    /** Get size, in GiB, of compressed (tensor train) representation. */
    auto GiB() const -> double { return to_GiB<T>(size()); }

    /** Get number of elements of uncompressed representation. */
    auto uncompressed_size() const -> size_type {
        return std::accumulate(
            modes_.cbegin(), modes_.cend(), 1, std::multiplies<size_type>());
    }

    /** Get size, in GiB, of uncompressed representation. */
    auto uncompressed_GiB() const -> double {
        return to_GiB<T>(uncompressed_size());
    }

    /** Get compression rate. */
    auto compression() const -> double {
        return (1.0 - static_cast<double>(size()) /
                          static_cast<double>(uncompressed_size()));
    }
    /*\@}*/
    /** Reconstruct full tensor from `*this` tensor train. */
    auto to_full() const -> Eigen::Tensor<T, D> {
        Eigen::Tensor<T, D> full(modes_);

        extent_type<1> c_axes = {index_pair_type(1, 0)};
        // contract dimension 1 of first chipped core with dimension 0 of second core
        Eigen::Tensor<T, 3> post =
            (cores_[0].chip(0, 0)).contract(cores_[1], c_axes);

        // recursion
        to_full<2>(post, full);

        return full;
    }

    template <typename U, typename V = typename std::common_type<T, U>::type>
    auto inner_product(TensorTrain<U, D> &Y) -> V {
        using core = Eigen::Tensor<V, 3>;

        // the W matrices have dimension \f$R_{n}^{Y} \times R_{n}^{X}\f$
        // the first such matrix is the identity
        Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> Id;
        Id.setIdentity(Y.rank(0), this->rank(0));
        // we handle them as 2-mode tensors, because it's more convenient to use the
        // contraction API (and let Eigen decide how to dispatch them to GEMMs)
        using matrix = Eigen::Tensor<V, 2>;
        matrix W = Eigen::TensorMap<matrix>(Id.data(), Id.rows(), Id.cols());

        for (auto i = 0; i < D; ++i) {
            // define core of auxiliary tensor Z with shape
            // \f$\lbrace R_{n}^{Y}, I_{n}, R_{n+1}^{X} \rbrace\f$
            // this is the mode-0 product of the i-th core with the W matrix
            extent_type<1> c_axes_Z = {index_pair_type(0, 1)};
            // in Eigen, contractions only need contraction axes as input: the
            // uncontracted axes might be laid out arbitrarily. We thus need a
            // shuffle after contraction to dimension the auxiliary core properly.
            Eigen::array<size_type, 3> shuffle = {2, 0, 1};
            core Z = cores_[i].contract(W, c_axes_Z).shuffle(shuffle);

            // update the W matrix as the mode-0 and mode-2 contraction of the Y and
            // Z cores
            // this is equivalent to the product: V_Y^t * V_Z, where V means vertical
            // unfolding
            extent_type<2> c_axes_W = {index_pair_type(0, 0), index_pair_type(1, 1)};
            W = Y.core(i).contract(Z, c_axes_W);
        }

        // the value of the inner product is the only element of the final 1x1 W
        // matrix
        return W(0);
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
    if (X.modes() != Y.modes()) {
        SPDLOG_ERROR(
            "X and Y tensors not compatible! X.modes() ({}) != Y.modes() ({}))",
            X.modes(),
            Y.modes());
        std::abort();
    }

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
auto sum(const TensorTrain<T, D> &X, const TensorTrain<U, D> &Y, double epsilon)
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
    if (X.modes() != Y.modes()) {
        SPDLOG_ERROR(
            "X and Y tensors not compatible! X.modes() ({}) != Y.modes() ({}))",
            X.modes(),
            Y.modes());
        std::abort();
    }

    // build up the shapes of the tensor cores for the sum
    // the ranks are the sum of those of X and Y, except for first and last
    std::array<size_type, D + 1> ranks;
    for (auto i = 0; i < D + 1; ++i) { ranks[i] = X.rank(i) * Y.rank(i); }

    TensorTrain<V, D> Z(X.modes(), ranks);

    // compute cores as the tensor product of the slices

    using index_pair_type = typename TensorTrain<V, D>::index_pair_type;
    using extent_type = typename TensorTrain<V, D>::template extent_type<1>;

    extent_type c_axes = {index_pair_type(1, 1)};

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
                    .contract(Y.core(i).slice(offs, ext_Y), c_axes)
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
                      double epsilon) -> TensorTrain<V, D> {
    auto Z = hadamard_product(X, Y);

    // perform rounding.
    Z.round(epsilon);

    return Z;
}
} // namespace tteigen
