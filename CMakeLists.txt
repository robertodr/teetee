cmake_minimum_required(VERSION 3.19)

project(teetee LANGUAGES CXX C)

find_package(LAPACK REQUIRED)

#add_subdirectory(external/upstream)
# all the external modules
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_eigen3.cmake)
# xtensor stack: order is important
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_xtl.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_xsimd.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_xtensor.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_xtensor-blas.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_HighFive.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/external/upstream/fetch_xtensor-io.cmake)

add_executable(foo foo.cpp)

target_compile_features(foo
  PRIVATE
    cxx_std_17
  )

target_compile_options(foo
  PRIVATE
    "-march=native"
  )

target_compile_definitions(foo
  PRIVATE
    XTENSOR_USE_XSIMD
  )

target_link_libraries(foo 
  PRIVATE 
    Eigen3::Eigen 
    xtl 
    xsimd 
    xtensor 
    xtensor-blas
    xtensor-io
    HighFive
    LAPACK::LAPACK
  )
