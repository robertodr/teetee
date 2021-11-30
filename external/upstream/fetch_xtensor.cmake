set(xtensor_pinned "0.24.0")

find_package(xtensor ${xtensor_pinned} CONFIG QUIET
  NO_CMAKE_PATH
  NO_CMAKE_PACKAGE_REGISTRY
  )

if(TARGET xtensor)
  message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIR} (found version ${xtensor_VERSION})")
else()
  message(STATUS "Suitable xtensor could not be located. Fetching!")
  include(FetchContent)

  FetchContent_Declare(xtensor
    QUIET
    URL
      https://github.com/QuantStack/xtensor/archive/${xtensor_pinned}.zip
    )

  set(xtl_DIR ${FETCHCONTENT_BASE_DIR}/xtl-build CACHE STRING "" FORCE)
  set(xsimd_DIR ${FETCHCONTENT_BASE_DIR}/xsimd-build CACHE STRING "" FORCE)
  set(XTENSOR_USE_XSIMD ON CACHE BOOL "" FORCE)
  set(CPP17 ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(xtensor)
endif()
