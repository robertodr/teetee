#include <cstdlib>

#include "tteigen.hpp"
#include "utils.hpp"

int main() {
    auto A = tteigen::sample_tensor();

    auto eig_tt = tteigen::tt_svd(A);

    return EXIT_SUCCESS;
}
