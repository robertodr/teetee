find_package(OpenMP 4.5 REQUIRED CXX)

find_package(LAPACK REQUIRED)

add_library(teetee INTERFACE)

target_compile_features(teetee
  INTERFACE
    cxx_std_17
  )

target_compile_options(teetee
  INTERFACE
    "-march=native"
  )

target_compile_definitions(teetee
  INTERFACE
    # make xtensor use xsimd
    XTENSOR_USE_XSIMD
    # WARNING if I use these with MKL, I get different SVDs
    # however, the tests still pass.
    # make Eigen use BLAS and LAPACK libraries
    EIGEN_USE_BLAS
    EIGEN_USE_LAPACKE_STRICT
    #EIGEN_USE_MKL_VML  # enables the use of Intel VML (vector operations)
    # compile HighFive to dump Eigen and xtensor objects to HDF5
    H5_USE_EIGEN
    H5_USE_XTENSOR
    # spdlog should use fmt installed with this project
    SPDLOG_FMT_EXTERNAL
  )

target_include_directories(teetee
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  )

target_precompile_headers(teetee
  INTERFACE
    # fmt
    "$<BUILD_INTERFACE:<fmt/core.h$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<fmt/chrono.h$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<fmt/ostream.h$<ANGLE-R>>"
    # spdlog
    "$<BUILD_INTERFACE:<spdlog/spdlog.h$<ANGLE-R>>"
    # xtensor
    #"$<BUILD_INTERFACE:<xtensor-blas/xlinalg.hpp$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<xtensor/xio.hpp$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<xtensor/xmath.hpp$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<xtensor/xtensor.hpp$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<xtensor/xview.hpp$<ANGLE-R>>"
    # Eigen
    "$<BUILD_INTERFACE:<Eigen/Core$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<Eigen/QR$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<Eigen/SVD$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<unsupported/Eigen/CXX11/Tensor$<ANGLE-R>>"
  )

target_link_libraries(teetee
  INTERFACE
    fmt::fmt
    OpenMP::OpenMP_CXX
    LAPACK::LAPACK
    HighFive
    Eigen3::Eigen
    xtl
    xsimd
    xtensor
    xtensor-blas
  )

add_executable(run_eigen run_eigen.cpp)

target_link_libraries(run_eigen
  PRIVATE
    teetee
  )
