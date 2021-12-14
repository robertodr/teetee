#pragma once

#include <array>
#include <cmath>
#include <numeric>

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
    return ((i0 / M_2_SQRTPI) - M_E * (i1 * 0.002) + std::pow(i2 + 1, -2) * M_PI * 0.03 - std::sin(i3 * M_PI_4) - 0.005 * std::cos(i5 / (i4 + 1)));
};
