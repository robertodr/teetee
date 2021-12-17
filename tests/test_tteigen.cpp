#include <cmath>

#include <catch2/catch.hpp>

#include <spdlog/spdlog.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include "eigen_utils.hpp"
#include "tteigen.hpp"

using namespace Catch::literals;

// warning works only for D = 6!
template <typename T>
inline Eigen::Tensor<T, 6> to_full(const tteigen::TensorTrain<T, 6> &tt) {
    // the reconstructed tensor
    Eigen::Tensor<T, 6> full;

    Eigen::array<Eigen::IndexPair<int>, 1> cdims = {Eigen::IndexPair<int>(1, 0)};
    // contract dimension 1 of first chipped core with dimension 0 of second core
    Eigen::Tensor<T, 3> tmp_1 =
        (tt.cores[0].chip(0, 0)).contract(tt.cores[1], cdims);

    cdims = {Eigen::IndexPair<int>(2, 0)};
    // contract dimension 2 of temporary with dimension 0 of third core
    Eigen::Tensor<T, 4> tmp_2 = tmp_1.contract(tt.cores[2], cdims);

    cdims = {Eigen::IndexPair<int>(3, 0)};
    // contract dimension 3 of temporary with dimension 0 of fourth core
    Eigen::Tensor<T, 5> tmp_3 = tmp_2.contract(tt.cores[3], cdims);

    cdims = {Eigen::IndexPair<int>(4, 0)};
    // contract dimension 4 of temporary with dimension 0 of fifth core
    Eigen::Tensor<T, 6> tmp_4 = tmp_3.contract(tt.cores[4], cdims);

    cdims = {Eigen::IndexPair<int>(5, 0)};
    // contract dimension 5 of temporary with dimension 0 of fifth core, then chip
    full = tmp_4.contract(tt.cores[5], cdims).chip(0, 6);

    return full;
}

TEST_CASE("horizontal unfolding of 3-mode tensor",
          "[tt][eigen][unfold][horizontal]") {
    Eigen::Tensor<double, 3> A(3, 4, 2);

    auto v = 1.0;
    for (auto k = 0; k < A.dimension(2); ++k) {
        for (auto j = 0; j < A.dimension(1); ++j) {
            for (auto i = 0; i < A.dimension(0); ++i) {
                A(i, j, k) = v;
                v += 1.0;
            }
        }
    }

    Eigen::Matrix<double, 3, 8> ref;
    v = 1.0;
    for (auto j = 0; j < ref.cols(); ++j) {
        for (auto i = 0; i < ref.rows(); ++i) {
            ref(i, j) = v;
            v += 1.0;
        }
    }

    const auto unfold = tteigen::horizontal_unfolding(A);

    REQUIRE(allclose(unfold, ref));
}

TEST_CASE("vertical unfolding of 3-mode tensor", "[tt][eigen][unfold][vertical]") {
    Eigen::Tensor<double, 3> A(3, 4, 2);

    auto v = 1.0;
    for (auto k = 0; k < A.dimension(2); ++k) {
        for (auto j = 0; j < A.dimension(1); ++j) {
            for (auto i = 0; i < A.dimension(0); ++i) {
                A(i, j, k) = v;
                v += 1.0;
            }
        }
    }

    Eigen::Matrix<double, 12, 2> ref;
    v = 1.0;
    for (auto j = 0; j < ref.cols(); ++j) {
        for (auto i = 0; i < ref.rows(); ++i) {
            ref(i, j) = v;
            v += 1.0;
        }
    }

    const auto unfold = tteigen::vertical_unfolding(A);

    REQUIRE(allclose(unfold, ref));
}

TEST_CASE("tensor train format with SVD", "[tt][eigen][svd]") {
    auto A = tteigen::sample_tensor();

    // take a copy so we can do elementwise comparison
    const Eigen::Tensor<double, 6> B = A;

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::tt_svd(A, epsilon);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = to_full(tt_A);

    const Eigen::Tensor<double, 0> B_norm = B.square().sum().sqrt();
    const double B_F = B_norm.coeff();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("left-multiplication by a scalar", "[tt][eigen][scalar-left-multiply]") {
    auto A = tteigen::sample_tensor();

    // take a copy so we can do elementwise comparison
    const Eigen::Tensor<double, 6> B = 2.5 * A;

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::tt_svd(A, epsilon);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = to_full(2.5 * tt_A);

    const Eigen::Tensor<double, 0> B_norm = B.square().sum().sqrt();
    const double B_F = B_norm.coeff();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("right-multiplication by a scalar", "[tt][eigen][scalar-right-multiply]") {
    auto A = tteigen::sample_tensor();

    // take a copy so we can do elementwise comparison
    const Eigen::Tensor<double, 6> B = A * 2.5;

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::tt_svd(A, epsilon);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = to_full(tt_A * 2.5);

    const Eigen::Tensor<double, 0> B_norm = B.square().sum().sqrt();
    const double B_F = B_norm.coeff();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("sum of two tensor trains, without rounding",
          "[tt][eigen][sum][no-rounding]") {
    auto A = tteigen::sample_tensor();

    // take a copy so we can do elementwise comparison
    const Eigen::Tensor<double, 6> B = A + A;

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::tt_svd(A, epsilon);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = to_full(tt_A + tt_A);

    const Eigen::Tensor<double, 0> B_norm = B.square().sum().sqrt();
    const double B_F = B_norm.coeff();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

TEST_CASE("Hadamard (elementwise) product of two tensor trains, without rounding",
          "[tt][eigen][hadamard][no-rounding]") {
    auto A = tteigen::sample_tensor();

    // take a copy so we can do elementwise comparison
    const Eigen::Tensor<double, 6> B = A * A;

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::tt_svd(A, epsilon);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = to_full(tteigen::hadamard_product(tt_A, tt_A));

    const Eigen::Tensor<double, 0> B_norm = B.square().sum().sqrt();
    const double B_F = B_norm.coeff();

    Eigen::Tensor<double, 0> tmp = (check - B).square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm <= epsilon * B_F);

    REQUIRE(allclose(check, B, 0.0, 1e-12));
}

/*
// scalar product of two tensors
// norm
*/
