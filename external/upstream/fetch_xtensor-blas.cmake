set(xtensor-blas_pinned "0.20.0")

find_package(xtensor-blas ${xtensor-blas_pinned} CONFIG QUIET
  NO_CMAKE_PATH
  NO_CMAKE_PACKAGE_REGISTRY
  )

if(TARGET xtensor-blas)
  message(STATUS "Found xtensor-blas: ${xtensor-blas_INCLUDE_DIR} (found version ${xtensor-blas_VERSION})")
else()
  message(STATUS "Suitable xtensor-blas could not be located. Fetching!")
  include(FetchContent)

  FetchContent_Declare(xtensor-blas
    QUIET
    URL
      https://github.com/QuantStack/xtensor-blas/archive/${xtensor-blas_pinned}.zip
    )

  set(xtensor_DIR ${FETCHCONTENT_BASE_DIR}/xtensor-build CACHE STRING "" FORCE)
  FetchContent_MakeAvailable(xtensor-blas)
endif()
