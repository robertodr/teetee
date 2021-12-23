#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

template <typename DerivedA, typename DerivedB>
bool allclose(const Eigen::DenseBase<DerivedA> &a,
              const Eigen::DenseBase<DerivedB> &b,
              const typename DerivedA::RealScalar &rtol = 1e-5,
              const typename DerivedA::RealScalar &atol = 1e-8) {
    return ((a.derived() - b.derived()).array().abs() <=
            (atol + rtol * b.derived().array().abs()))
        .all();
}

template <typename T, int D>
inline auto allclose(const Eigen::Tensor<T, D> &a,
                     const Eigen::Tensor<T, D> &b,
                     T rtol = 1e-5,
                     T atol = 1e-8) -> bool {
    Eigen::Tensor<bool, 0> tmp = ((a - b).abs() <= (atol + rtol * b.abs())).all();
    return tmp.coeff();
}
