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
    # WARNING if I use these with MKL, I get different SVDs
    # however, the tests still pass.
    # make Eigen use BLAS and LAPACK libraries
    EIGEN_USE_BLAS
    EIGEN_USE_LAPACKE_STRICT
    # spdlog should use fmt installed with this project
    SPDLOG_FMT_EXTERNAL
    # tell HighFive to enable the Eigen backend
    H5_USE_EIGEN
  )

# TODO find a way to enable this only if we're using MKL
#target_compile_definitions(teetee
#  INTERFACE
#    # enables the use of Intel VML (vector operations)
#    EIGEN_USE_MKL_VML
#  )

target_include_directories(teetee
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  )

target_precompile_headers(teetee
  INTERFACE
    "$<BUILD_INTERFACE:<Eigen/Core$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<Eigen/QR$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<Eigen/SVD$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<unsupported/Eigen/CXX11/Tensor$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<fmt/core.h$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<fmt/chrono.h$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<fmt/ostream.h$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<fmt/ranges.h$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<spdlog/spdlog.h$<ANGLE-R>>"
    "$<BUILD_INTERFACE:<highfive/H5Easy.hpp$<ANGLE-R>>"
  )

target_link_libraries(teetee
  INTERFACE
    fmt::fmt
    OpenMP::OpenMP_CXX
    LAPACK::LAPACK
    HighFive
    Eigen3::Eigen
  )

foreach(_exe run_eigen eigensvd generate_svd_references)
  add_executable(${_exe} ${_exe}.cpp)

  target_link_libraries(${_exe}
    PRIVATE
      teetee
    )
endforeach()
