#include <cstdlib>

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "eigen_utils.hpp"
#include "tteigen.hpp"
#include "utils.hpp"

int main() {
    spdlog::set_pattern("[%Y-%m-%d %T][%^%l%$][TID: %t, PID: %P][%!@%s:%4#] %v");

    auto A = tteigen::sample_tensor<5, 5 ,5, 5, 5, 5>();

    SPDLOG_INFO("size of tensor = {} elements", A.size());
    SPDLOG_INFO("memory ~ {:.2e} GiB", to_GiB<double>(A.size()));

    auto start = std::chrono::steady_clock::now();

    auto tt_A = tteigen::tt_svd(A);

    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;

    SPDLOG_INFO("decomposition with Eigen done in {}", elapsed);

    auto ncore = 0;
    for (const auto &c : tt_A.cores) {
        SPDLOG_INFO("core {} with shape = {}", ncore, c.dimensions());
        ncore += 1;
    }
    SPDLOG_INFO("ranks {}", tt_A.ranks);
    SPDLOG_INFO("size of TT format = {} elements", tt_A.size());
    SPDLOG_INFO("memory ~ {:.2e} GiB", to_GiB<double>(tt_A.size()));

    auto compression =
        (1.0 - static_cast<double>(tt_A.size()) / static_cast<double>(A.size()));
    SPDLOG_INFO("compression = {:2.2f}%", compression * 100);

    auto Y = tteigen::right_orthonormalize(tt_A);

    SPDLOG_INFO("norm {}", Y.norm());

    return EXIT_SUCCESS;
}
