#include <catch2/catch.hpp>

#include <unsupported/Eigen/CXX11/Tensor>

#include "tteigen.hpp"
#include "utils.hpp"

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

TEST_CASE("TT-SVD with Eigen", "[tt-svd][eigen]") {
    auto A = tteigen::sample_tensor();

    // norm of tensor --> gives us the threshold for the SVDs
    const Eigen::Tensor<double, 0> A_norm = A.square().sum().sqrt();
    const double A_F = A_norm.coeff();

    const auto epsilon = 1.0e-12;
    auto tt_A = tteigen::tt_svd(A, epsilon);

    // the reconstructed tensor
    Eigen::Tensor<double, 6> check = to_full(tt_A);

    Eigen::Tensor<double, 0> tmp = check.square().sum().sqrt();
    const double check_norm = tmp.coeff();

    REQUIRE(check_norm == Approx(A_F));
}

/*
// scalar times TT
checkA = to_full(2.5 * tt);
foo = (2.5 * A - checkA).square().sum().sqrt();
check_norm = foo.coeff();
std::cout << "CHECK: scalar times TT" << std::endl;
std::cout << "Norm of difference = " << check_norm << std::endl;
if (std::abs(check_norm) >= 2.5 * epsilon * A_F) {
    std::cout << "TT format not within tolerance!" << std::endl;
} else {
    std::cout << "All good!" << std::endl;
}

// TT times scalar
checkA = to_full(tt * 2.5);
foo = (2.5 * A - checkA).square().sum().sqrt();
check_norm = foo.coeff();
std::cout << "CHECK: TT times scalar" << std::endl;
std::cout << "Norm of difference = " << check_norm << std::endl;
if (std::abs(check_norm) >= 2.5 * epsilon * A_F) {
    std::cout << "TT format not within tolerance!" << std::endl;
} else {
    std::cout << "All good!" << std::endl;
}

// sum of two tensors
checkA = to_full(tt + tt);
foo = (2.0 * A - checkA).square().sum().sqrt();
check_norm = foo.coeff();
std::cout << "CHECK: TT plus TT (no rounding)" << std::endl;
std::cout << "Norm of difference = " << check_norm << std::endl;
if (std::abs(check_norm) >= 2.0 * epsilon * A_F) {
    std::cout << "TT format not within tolerance!" << std::endl;
} else {
    std::cout << "All good!" << std::endl;
}

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
