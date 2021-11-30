#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

// xtensor MUST be included before Eigen!
//#include <xtensor/xtensor.hpp>

#include <unsupported/Eigen/CXX11/Tensor>

template <typename T> auto stream_collection(const T &coll) -> std::string {
    std::ostringstream os;
    bool first = true;
    os << "[";
    for (auto elem : coll) {
        if (!first) os << ", ";
        os << elem;
        first = false;
    }
    os << "]";
    return os.str();
}

template <typename T, std::size_t D> auto operator<<(std::ostream &os, const std::array<T, D> &coll) -> std::ostream & {
    return (os << stream_collection(coll));
}

template <std::size_t N> struct RowMajor final {
    inline std::size_t operator()(const std::array<std::size_t, N> &idxs) { return std::inner_product(idxs.cbegin(), idxs.cend(), coefs.cbegin(), 0); }

    RowMajor(const std::array<std::size_t, N> &dims) {
        coefs.back() = 1;
        std::partial_sum(dims.cbegin() + 1, dims.cend(), coefs.begin(), [](std::size_t a, std::size_t b) -> std::size_t { return a * b; });
    }

    std::array<std::size_t, N> coefs;
};

template <std::size_t N> struct ColMajor final {
    inline std::size_t operator()(const std::array<std::size_t, N> &idxs) { return std::inner_product(idxs.cbegin(), idxs.cend(), coefs.cbegin(), 0); }

    ColMajor(const std::array<std::size_t, N> &dims) {
        coefs.front() = 1;
        std::partial_sum(dims.cbegin(), dims.cend() - 1, coefs.begin() + 1, [](std::size_t a, std::size_t b) -> std::size_t { return a * b; });
    }

    std::array<std::size_t, N> coefs;
};

template <typename T> auto to_GiB(std::size_t count) -> double {
    return (count * sizeof(T) / (1024.0 * 1024.0 * 1024.0));
}

auto value(int i0, int i1, int i2, int i3, int i4, int i5) -> double {
    return 1.0e-6 * ((i0 / M_2_SQRTPI) - M_E * (i1 * 0.002) + std::pow(i2 + 1, -2) * M_PI * 0.03 - std::sin(i3 * M_PI_4) - 0.005 * std::cos(i5 / (i4 + 1)));
};

namespace tteigen {
auto sample_tensor() -> Eigen::Tensor<double, 6> {
    constexpr auto d0 = 5;
    constexpr auto d1 = 5;
    constexpr auto d2 = 5;
    constexpr auto d3 = 5;
    constexpr auto d4 = 5;
    constexpr auto d5 = 5;

    ColMajor colmajor_linear_id(std::array<std::size_t, 6>{d0, d1, d2, d3, d4, d5});

    auto buffer = static_cast<double *>(std::aligned_alloc(alignof(double), sizeof(double) * d0 * d1 * d2 * d3 * d4 * d5));

    // fill
    for (std::size_t i0 = 0; i0 < d0; ++i0) {
        for (std::size_t i1 = 0; i1 < d1; ++i1) {
            for (std::size_t i2 = 0; i2 < d2; ++i2) {
                for (std::size_t i3 = 0; i3 < d3; ++i3) {
                    for (std::size_t i4 = 0; i4 < d4; ++i4) {
                        for (std::size_t i5 = 0; i5 < d5; ++i5) {
                            auto idx = colmajor_linear_id({i0, i1, i2, i3, i4, i5});
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
