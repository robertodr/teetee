#include <algorithm>
#include <cstdlib>
#include <random>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <highfive/H5Easy.hpp>

int main() {
    H5Easy::File file("svd_bench.h5", H5Easy::File::Overwrite);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // initialize Eigen Jacobi SVD with preconditioner FullPivHouseholderQRPreconditioner
    Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::FullPivHouseholderQRPreconditioner> svd;

    constexpr auto sz = 2683044;
    // square matrix
    auto n_rows = 1638;
    auto n_cols = sz / n_rows;
    std::vector<double> a(n_rows * n_cols);
    std::generate(a.begin(), a.end(), [&dist, &mt]() { return dist(mt); });
    H5Easy::dump(file, "/raw/a", a);

    Eigen::Map<Eigen::MatrixXd> A(a.data(), n_rows, n_cols);

    auto A_svd = svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    H5Easy::dump(file, "/ref/s_A", A_svd.singularValues());

    // broad rectangular matrix
    n_rows = 546;
    n_cols = sz / n_rows;
    std::vector<double> b(n_rows * n_cols);
    std::generate(b.begin(), b.end(), [&dist, &mt]() { return dist(mt); });
    H5Easy::dump(file, "/raw/b", b);

    Eigen::Map<Eigen::MatrixXd> B(b.data(), n_rows, n_cols);

    auto B_svd = svd.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    H5Easy::dump(file, "/ref/s_B", B_svd.singularValues());

    // skinny rectangular matrix (swap dimensions of the broad matrix)
    std::vector<double> c(n_rows * n_cols);
    std::generate(c.begin(), c.end(), [&dist, &mt]() { return dist(mt); });
    H5Easy::dump(file, "/raw/c", c);

    Eigen::Map<Eigen::MatrixXd> C(c.data(), n_cols, n_rows);

    auto C_svd = svd.compute(C, Eigen::ComputeThinU | Eigen::ComputeThinV);

    H5Easy::dump(file, "/ref/s_C", C_svd.singularValues());

    return EXIT_SUCCESS;
}
