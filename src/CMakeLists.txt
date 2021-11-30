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
    XTENSOR_USE_XSIMD
    # WARNING if I use these with MKL, I get different SVDs
    # however, the tests still pass.
    EIGEN_USE_BLAS
    EIGEN_USE_LAPACKE_STRICT
    #EIGEN_USE_MKL_VML  # enables the use of Intel VML (vector operations)
    H5_USE_EIGEN
    H5_USE_XTENSOR
  )

target_include_directories(teetee
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  )

target_link_libraries(teetee
  INTERFACE
    OpenMP::OpenMP_CXX
    LAPACK::LAPACK
    HighFive
    Eigen3::Eigen
    xtl
    xsimd
    xtensor
    xtensor-blas
  )

add_executable(run_teetee main.cpp)

target_link_libraries(run_teetee
  PRIVATE
    teetee
  )
