#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "utils.hpp"

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

namespace tteigen {
template <typename Indexing = ColMajor<6>,
          std::size_t d0 = 5,
          std::size_t d1 = 5,
          std::size_t d2 = 5,
          std::size_t d3 = 5,
          std::size_t d4 = 5,
          std::size_t d5 = 5>
auto sample_tensor() -> Eigen::Tensor<double, 6> {
    Indexing linear_id(std::array<std::size_t, 6>{d0, d1, d2, d3, d4, d5});

    auto count = d0 * d1 * d2 * d3 * d4 * d5;
    auto buffer = static_cast<double *>(
        std::aligned_alloc(alignof(double), sizeof(double) * count));

    // fill
    for (std::size_t i0 = 0; i0 < d0; ++i0) {
        for (std::size_t i1 = 0; i1 < d1; ++i1) {
            for (std::size_t i2 = 0; i2 < d2; ++i2) {
                for (std::size_t i3 = 0; i3 < d3; ++i3) {
                    for (std::size_t i4 = 0; i4 < d4; ++i4) {
                        for (std::size_t i5 = 0; i5 < d5; ++i5) {
                            auto idx = linear_id({i0, i1, i2, i3, i4, i5});
                            buffer[idx] = value(i0, i1, i2, i3, i4, i5);
                        }
                    }
                }
            }
        }
    }

    Eigen::TensorMap<Eigen::Tensor<double, 6>> A(buffer, d0, d1, d2, d3, d4, d5);

    return A;
}
} // namespace tteigen
