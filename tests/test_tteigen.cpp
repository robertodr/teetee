#include <cmath>

#include <catch2/catch.hpp>

#include <spdlog/spdlog.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include "eigen_utils.hpp"
#include "test_utils.hpp"
#include "tteigen.hpp"
#include "utils.hpp"

using namespace tteigen;

TEST_CASE("tensor train format with SVD", "[tt][eigen][svd]") {
    auto A = sample_tensor<5, 5, 5, 5, 5, 5>();
    double A_F = frobenius_norm(A.data(), A.size());

    auto epsilon = 1.0e-12;
    auto tt_A = TensorTrain(A, epsilon);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - A).square().sum().sqrt();
    double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * A_F);

    REQUIRE(allclose(check, A, 0.0, 1e-12));
}

TEST_CASE("right-to-left orthogonalization of tensor train, with thin QR",
          "[tt][eigen][orthogonalize-RL][thin-QR]") {
    using shape_type = typename TensorTrain<double, 6>::shape_type;
    using core_type = typename TensorTrain<double, 6>::core_type;
    std::array<core_type, 6> cores = {core_type(shape_type{1, 5, 4}).setRandom(),
                                      core_type(shape_type{4, 8, 2}).setRandom(),
                                      core_type(shape_type{2, 5, 3}).setRandom(),
                                      core_type(shape_type{3, 6, 4}).setRandom(),
                                      core_type(shape_type{4, 6, 4}).setRandom(),
                                      core_type(shape_type{4, 10, 1}).setRandom()};

    auto tt_A = TensorTrain<double, 6>(cores);

    // reconstruct full tensor *before* orthogonalization
    Eigen::Tensor<double, 6> A = tt_A.to_full();

    tt_A.orthogonalize_RL();

    // test that all horizontal unfoldings of cores >= 1 are row orthonormal
    using matrix_type = typename TensorTrain<double, 6>::matrix_type;
    for (auto i = 1; i < 6; ++i) {
        auto c = tt_A.core(i);
        // horizontal unfolding
        auto n_rows = c.dimension(0);
        auto n_cols = c.dimension(1) * c.dimension(2);
        matrix_type H = Eigen::Map<matrix_type>(c.data(), n_rows, n_cols);

        REQUIRE(allclose(H * H.adjoint(), matrix_type::Identity(n_rows, n_rows)));
    }

    // reconstruct full tensor *after* orthogonalization
    Eigen::Tensor<double, 6> orth_A = tt_A.to_full();

    REQUIRE(allclose(A, orth_A, 0.0, 1e-12));
}

TEST_CASE("right-to-left orthogonalization of tensor train, with regular QR",
          "[tt][eigen][orthogonalize-RL][regular-QR]") {
    using shape_type = typename TensorTrain<double, 6>::shape_type;
    using core_type = typename TensorTrain<double, 6>::core_type;
    std::array<core_type, 6> cores = {core_type(shape_type{1, 5, 4}).setRandom(),
                                      core_type(shape_type{4, 8, 2}).setRandom(),
                                      core_type(shape_type{2, 5, 3}).setRandom(),
                                      core_type(shape_type{3, 6, 4}).setRandom(),
                                      core_type(shape_type{4, 6, 4}).setRandom(),
                                      core_type(shape_type{25, 10, 1}).setRandom()};

    auto tt_A = TensorTrain<double, 6>(cores);

    // reconstruct full tensor *before* orthogonalization
    Eigen::Tensor<double, 6> A = tt_A.to_full();

    tt_A.orthogonalize_RL();

    // test that all horizontal unfoldings of cores >= 1 are row orthonormal
    using matrix_type = typename TensorTrain<double, 6>::matrix_type;
    for (auto i = 1; i < 6; ++i) {
        auto c = tt_A.core(i);
        // horizontal unfolding
        auto n_rows = c.dimension(0);
        auto n_cols = c.dimension(1) * c.dimension(2);
        matrix_type H = Eigen::Map<matrix_type>(c.data(), n_rows, n_cols);

        REQUIRE(allclose(H * H.adjoint(), matrix_type::Identity(n_rows, n_rows)));
    }

    // reconstruct full tensor *after* orthogonalization
    Eigen::Tensor<double, 6> orth_A = tt_A.to_full();

    REQUIRE(allclose(A, orth_A, 0.0, 1e-12));
}

TEST_CASE("rounding of tensor train", "[tt][eigen][rounding]") {
    auto A = sample_tensor<10, 9, 8, 7, 6, 5>();

    auto epsilon = 1.0e-18;
    auto tt_A = TensorTrain(A, epsilon);
    auto ranks_before = tt_A.ranks();

    epsilon = 1.0e-6;
    tt_A.round(epsilon);

    // check that the ranks were reduced
    // ranks are squared
    for (auto i = 0; i < 7; ++i) { REQUIRE(tt_A.rank(i) <= ranks_before[i]); }

    // reconstruct full tensor *after* rounding
    Eigen::Tensor<double, 6> round_A = tt_A.to_full();

    REQUIRE(allclose(A, round_A, 0.0, epsilon));
}

TEST_CASE("scaling", "[tt][eigen][scaling]") {
    auto A = sample_tensor<5, 5, 5, 5, 5, 5>();

    // scale by 2.5 as single-precision floating point
    auto alpha = 2.5f;

    // take a copy so we can do elementwise comparison
    Eigen::Tensor<double, 6> B = alpha * A;
    double B_F = frobenius_norm(B.data(), B.size());

    auto epsilon = 1.0e-12;
    auto tt_A = TensorTrain(A, epsilon);
    tt_A.scale(alpha);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("left-multiplication by a scalar", "[tt][eigen][scalar-left-multiply]") {
    auto A = sample_tensor<5, 5, 5, 5, 5, 5>();

    // scale by 2.5 as single-precision floating point
    auto alpha = 2.5f;

    // take a copy so we can do elementwise comparison
    Eigen::Tensor<double, 6> B = alpha * A;
    double B_F = frobenius_norm(B.data(), B.size());

    auto epsilon = 1.0e-12;
    auto tt_A = alpha * TensorTrain(A, epsilon);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("right-multiplication by a scalar", "[tt][eigen][scalar-right-multiply]") {
    auto A = sample_tensor<5, 5, 5, 5, 5, 5>();

    // scale by 2.5 as single-precision floating point
    auto alpha = 2.5f;

    // take a copy so we can do elementwise comparison
    Eigen::Tensor<double, 6> B = alpha * A;
    double B_F = frobenius_norm(B.data(), B.size());

    auto epsilon = 1.0e-12;
    auto tt_A = TensorTrain(A, epsilon) * alpha;

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("sum of two tensor trains, without rounding",
          "[tt][eigen][sum][no-rounding]") {
    auto A = sample_tensor<5, 5, 5, 5, 5, 5>();

    // take a copy so we can do elementwise comparison
    Eigen::Tensor<double, 6> B = A + A;
    double B_F = frobenius_norm(B.data(), B.size());

    auto epsilon = 1.0e-12;
    auto tt_A = TensorTrain(A, epsilon);

    auto tt_2A = sum(tt_A, tt_A);

    // ranks, except first and last, are doubled
    for (auto i = 1; i < 6; ++i) { REQUIRE(tt_2A.rank(i) == 2 * tt_A.rank(i)); }

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_2A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("sum of two tensor trains, with rounding", "[tt][eigen][sum][rounding]") {
    auto A = sample_tensor<5, 5, 5, 5, 5, 5>();

    // take a copy so we can do elementwise comparison
    Eigen::Tensor<double, 6> B = A + A;
    double B_F = frobenius_norm(B.data(), B.size());

    auto epsilon = 1.0e-12;
    auto tt_A = TensorTrain(A, epsilon);

    auto tt_2A = sum(tt_A, tt_A, epsilon);

    // after rounding, shapes should all be equal
    REQUIRE(tt_2A.shapes() == tt_A.shapes());

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_2A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("Hadamard (elementwise) product of two tensor trains, without rounding",
          "[tt][eigen][hadamard][no-rounding]") {
    auto A = sample_tensor<5, 5, 5, 5, 5, 5>();

    // take a copy so we can do elementwise comparison
    Eigen::Tensor<double, 6> B = A * A;
    double B_F = frobenius_norm(B.data(), B.size());

    auto epsilon = 1.0e-12;
    auto tt_A = TensorTrain(A, epsilon);

    auto had_A = hadamard_product(tt_A, tt_A);

    // ranks are squared
    for (auto i = 0; i < 7; ++i) {
        REQUIRE(had_A.rank(i) == tt_A.rank(i) * tt_A.rank(i));
    }

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = had_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("Hadamard (elementwise) product of two tensor trains, with rounding",
          "[tt][eigen][hadamard][rounding]") {
    auto A = sample_tensor<5, 5, 5, 5, 5, 5>();

    // take a copy so we can do elementwise comparison
    Eigen::Tensor<double, 6> B = A * A;
    double B_F = frobenius_norm(B.data(), B.size());

    auto epsilon = 1.0e-12;
    auto tt_A = TensorTrain(A, epsilon);

    auto had_A = hadamard_product(tt_A, tt_A, epsilon);

    // ranks are squared
    for (auto i = 0; i < 7; ++i) {
        REQUIRE(had_A.rank(i) <= tt_A.rank(i) * tt_A.rank(i));
    }

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = had_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("Frobenius norm of tensor train", "[tt][eigen][frobenius]") {
    auto A = sample_tensor<5, 5, 5, 5, 5, 5>();
    double A_F = frobenius_norm(A.data(), A.size());

    auto epsilon = 1.0e-12;
    auto tt_A = TensorTrain(A, epsilon);

    tt_A.orthogonalize_RL();

    REQUIRE(tt_A.norm() == Approx(A_F));
}

TEST_CASE("inner product of two tensor trains", "[tt][eigen][inner]") {
    auto A = sample_tensor<5, 5, 5, 5, 5, 5>();
    double A_F = frobenius_norm(A.data(), A.size());

    using index_pair_type = Eigen::IndexPair<Eigen::Index>;

    Eigen::array<index_pair_type, 6> cdims = {index_pair_type(0, 0),
                                              index_pair_type(1, 1),
                                              index_pair_type(2, 2),
                                              index_pair_type(3, 3),
                                              index_pair_type(4, 4),
                                              index_pair_type(5, 5)};
    Eigen::Tensor<double, 0> ref_dot = A.contract(A, cdims);

    auto epsilon = 1.0e-12;
    auto tt_A = TensorTrain(A, epsilon);

    auto dot = tt_A.inner_product(tt_A);

    REQUIRE(dot == Approx(ref_dot(0)));
}
