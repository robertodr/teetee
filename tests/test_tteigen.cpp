#include <cmath>

#include <catch2/catch.hpp>

#include <spdlog/spdlog.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include "eigen_utils.hpp"
#include "tteigen.hpp"
#include "utils.hpp"

TEST_CASE("tensor train format with SVD", "[tt][eigen][svd]") {
    const auto A = tteigen::sample_tensor<5, 5, 5, 5, 5, 5>();
    const double A_F = frobenius_norm(A.data(), A.size());

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::TT(A, epsilon);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - A).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * A_F);

    REQUIRE(allclose(check, A, 0.0, 1e-12));
}

TEST_CASE("scaling", "[tt][eigen][scaling]") {
    auto A = tteigen::sample_tensor<5, 5, 5, 5, 5, 5>();

    // scale by 2.5 as single-precision floating point
    auto alpha = 2.5f;

    // take a copy so we can do elementwise comparison
    const Eigen::Tensor<double, 6> B = alpha * A;
    const double B_F = frobenius_norm(B.data(), B.size());

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::TT(A, epsilon);
    tt_A.scale(alpha);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("left-multiplication by a scalar", "[tt][eigen][scalar-left-multiply]") {
    auto A = tteigen::sample_tensor<5, 5, 5, 5, 5, 5>();

    // scale by 2.5 as single-precision floating point
    auto alpha = 2.5f;

    // take a copy so we can do elementwise comparison
    const Eigen::Tensor<double, 6> B = alpha * A;
    const double B_F = frobenius_norm(B.data(), B.size());

    const auto epsilon = 1.0e-12;
    auto tt_A = alpha * tteigen::TT(A, epsilon);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("right-multiplication by a scalar", "[tt][eigen][scalar-right-multiply]") {
    auto A = tteigen::sample_tensor<5, 5, 5, 5, 5, 5>();

    // scale by 2.5 as single-precision floating point
    auto alpha = 2.5f;

    // take a copy so we can do elementwise comparison
    const Eigen::Tensor<double, 6> B = alpha * A;
    const double B_F = frobenius_norm(B.data(), B.size());

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::TT(A, epsilon) * alpha;

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("sum of two tensor trains, without rounding",
          "[tt][eigen][sum][no-rounding]") {
    auto A = tteigen::sample_tensor<5, 5, 5, 5, 5, 5>();

    // take a copy so we can do elementwise comparison
    const Eigen::Tensor<double, 6> B = A + A;
    const double B_F = frobenius_norm(B.data(), B.size());

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::TT(A, epsilon);

    auto tt_2A = sum(tt_A, tt_A);
    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_2A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("Hadamard (elementwise) product of two tensor trains, without rounding",
          "[tt][eigen][hadamard][no-rounding]") {
    auto A = tteigen::sample_tensor<5, 5, 5, 5, 5, 5>();

    // take a copy so we can do elementwise comparison
    const Eigen::Tensor<double, 6> B = A * A;
    const double B_F = frobenius_norm(B.data(), B.size());

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::TT(A, epsilon);

    auto had_A = tteigen::hadamard_product(tt_A, tt_A);
    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = had_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("right orthonormalization of tensor train",
          "[tt][eigen][right-orthonormalization]") {
    const auto A = tteigen::sample_tensor<5, 5, 5, 5, 5, 5>();
    const double A_F = frobenius_norm(A.data(), A.size());

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::TT(A, epsilon);

    tt_A.right_orthonormalize();

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = tt_A.to_full();

    Eigen::Tensor<double, 0> tmp = (check - A).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * A_F);

    REQUIRE(allclose(check, A, 0.0, 1e-12));

    // FIXME also test that all the cores are row orthonormal!
}

TEST_CASE("Frobenius norm of tensor train", "[tt][eigen][frobenius]") {
    const auto A = tteigen::sample_tensor<5, 5, 5, 5, 5, 5>();
    const double A_F = frobenius_norm(A.data(), A.size());

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::TT(A, epsilon);

    tt_A.right_orthonormalize();

    REQUIRE(tt_A.norm() == Approx(A_F));
}

/*
// scalar product of two tensors
*/
