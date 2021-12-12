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
    XTENSOR_DEFAULT_LAYOUT=::xt::layout_type::column_major
    # make xtensor-blas use detected BLAS/LAPACK
    HAVE_CBLAS=1
    # WARNING if I use these with MKL, I get different SVDs
    # however, the tests still pass.
    # make Eigen use BLAS and LAPACK libraries
    EIGEN_USE_BLAS
    EIGEN_USE_LAPACKE_STRICT
    # spdlog should use fmt installed with this project
    SPDLOG_FMT_EXTERNAL
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

foreach(_exe run_eigen run_xtensor eigensvd xtsvd generate_svd_references)
  add_executable(${_exe} ${_exe}.cpp)

  target_link_libraries(${_exe}
    PRIVATE
      teetee
    )
endforeach()
