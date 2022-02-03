#include <cstdlib>

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "eigen_utils.hpp"
#include "tteigen.hpp"
#include "utils.hpp"

using namespace tteigen;

int main() {
    spdlog::set_pattern("[%Y-%m-%d %T][%^%l%$][TID: %t, PID: %P][%!@%s:%4#] %v");

    const auto A = sample_tensor<25, 25, 5, 10, 12, 18>();

    SPDLOG_INFO("size of tensor = {} elements", A.size());
    SPDLOG_INFO("memory ~ {:.2e} GiB", to_GiB<double>(A.size()));

    auto start = std::chrono::steady_clock::now();

    auto tt_A = TensorTrain(A, 1.e-8);

    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;

    SPDLOG_INFO("decomposition with Eigen done in {}", elapsed);

    SPDLOG_INFO("ranks {}", tt_A.ranks());
    SPDLOG_INFO("size of TT format = {} elements", tt_A.size());
    SPDLOG_INFO("memory ~ {:.2e} GiB", tt_A.GiB());

    auto compression = tt_A.compression();
    SPDLOG_INFO("compression = {:2.2f}%", compression * 100);

    auto rando_A = TensorTrain(A);

    SPDLOG_INFO("ranks {}", rando_A.ranks());
    SPDLOG_INFO("size of TT format = {} elements", rando_A.size());
    SPDLOG_INFO("memory ~ {:.2e} GiB", rando_A.GiB());

    compression = rando_A.compression();
    SPDLOG_INFO("compression = {:2.2f}%", compression * 100);

    return EXIT_SUCCESS;
}
