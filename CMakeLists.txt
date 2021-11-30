cmake_minimum_required(VERSION 3.19)

project(teetee LANGUAGES CXX C)

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# all the external modules
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_eigen3.cmake)
# xtensor stack: order is important
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_xtl.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_xsimd.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_xtensor.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_xtensor-blas.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_HighFive.cmake)

add_subdirectory(src)

enable_testing()

add_subdirectory(tests)
