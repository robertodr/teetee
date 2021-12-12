#include <catch2/catch.hpp>

#include <unsupported/Eigen/CXX11/Tensor>

#include "eigen_utils.hpp"
#include "tteigen.hpp"

using namespace Catch::literals;

// warning works only for D = 6!
template <typename T> Eigen::Tensor<T, 6> to_full(const tteigen::TensorTrain<T, 6> &tt) {
    // the reconstructed tensor
    Eigen::Tensor<T, 6> full;

    Eigen::array<Eigen::IndexPair<int>, 1> cdims = {Eigen::IndexPair<int>(1, 0)};
    // contract dimension 1 of first chipped core with dimension 0 of second core
    Eigen::Tensor<T, 3> tmp_1 = (tt.cores[0].chip(0, 0)).contract(tt.cores[1], cdims);

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

TEST_CASE("Eigen :: unfoldings of 3-mode tensor", "[tt][eigen][unfold]") {
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

    SECTION("horizontal (mode-0) unfolding") {
        const auto unfold = tteigen::H_unfold(A);

        Eigen::Map<Eigen::MatrixXd> ref(A.data(), A.dimension(0), A.size() / A.dimension(0));

        REQUIRE(unfold.isApprox(ref));
    }

    SECTION("mode-1 unfolding") {
        const auto unfold = tteigen::unfold(1, A);

        Eigen::Map<Eigen::MatrixXd> ref(A.data(), A.dimension(1), A.size() / A.dimension(1));

        REQUIRE(unfold.isApprox(ref));
    }

    SECTION("vertical (mode-2) unfolding") {
        const auto unfold = tteigen::V_unfold(A);

        Eigen::Map<Eigen::MatrixXd> ref(A.data(), A.dimension(2), A.size() / A.dimension(2));

        REQUIRE(unfold.isApprox(ref));
    }
}

TEST_CASE("Eigen :: tensor train format", "[tt][eigen]") {
    auto A = tteigen::sample_tensor();

    // norm of tensor --> gives us the threshold for the SVDs
    const Eigen::Tensor<double, 0> A_norm = A.square().sum().sqrt();
    const double A_F = A_norm.coeff();

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::tt_svd(A, epsilon);

    SECTION("decomposition") {
        // the reconstructed tensor
        Eigen::Tensor<double, 6> check = to_full(tt_A);

        Eigen::Tensor<double, 0> tmp = check.square().sum().sqrt();
        const double check_norm = tmp.coeff();

        REQUIRE(check_norm == Approx(A_F));
    }

    SECTION("multiplication by a scalar from the left") {
        // the reconstructed tensor
        Eigen::Tensor<double, 6> check = to_full(2.5 * tt_A);

        Eigen::Tensor<double, 0> tmp = check.square().sum().sqrt();
        const double check_norm = tmp.coeff();

        REQUIRE(check_norm == Approx(2.5 * A_F));
    }

    SECTION("multiplication by a scalar from the right") {
        // the reconstructed tensor
        Eigen::Tensor<double, 6> check = to_full(tt_A * 2.5);

        Eigen::Tensor<double, 0> tmp = check.square().sum().sqrt();
        const double check_norm = tmp.coeff();

        REQUIRE(check_norm == Approx(2.5 * A_F));
    }

    SECTION("sum of two tensor trains, without rounding") {
        // the reconstructed tensor
        Eigen::Tensor<double, 6> check = to_full(tt_A + tt_A);

        Eigen::Tensor<double, 0> tmp = check.square().sum().sqrt();
        const double check_norm = tmp.coeff();

        REQUIRE(check_norm == Approx(2.0 * A_F));
    }
}

/*
// Hadamard product of two tensors
checkA = to_full(hadamard_product(tt, tt));
foo = ((A * A) - checkA).square().sum().sqrt();
check_norm = foo.coeff();
std::cout << "CHECK: TT Hadamard-product TT (no rounding)" << std::endl;
std::cout << "Norm of difference = " << check_norm << std::endl;
if (std::abs(check_norm) >= std::pow(epsilon, 2) * A_F) {
    std::cout << "TT format not within tolerance!" << std::endl;
} else {
    std::cout << "All good!" << std::endl;
}

// scalar product of two tensors
// norm
*/
