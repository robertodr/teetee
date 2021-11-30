set(xtensor-io_pinned "0.13.0")

find_package(xtensor-io ${xtensor-io_pinned} CONFIG QUIET
  NO_CMAKE_PATH
  NO_CMAKE_PACKAGE_REGISTRY
  )

if(TARGET xtensor-io)
  message(STATUS "Found xtensor-io: ${xtensor-io_INCLUDE_DIR} (found version ${xtensor-io_VERSION})")
else()
  message(STATUS "Suitable xtensor-io could not be located. Fetching!")
  include(FetchContent)

  FetchContent_Declare(xtensor-io
    QUIET
    URL
      https://github.com/QuantStack/xtensor-io/archive/${xtensor-io_pinned}.zip
    )

  set(HAVE_HighFive ON CACHE BOOL "" FORCE)
  set(xtensor_DIR ${FETCHCONTENT_BASE_DIR}/xtensor-build CACHE STRING "" FORCE)
  set(HighFive_DIR ${FETCHCONTENT_BASE_DIR}/highfive-build CACHE STRING "" FORCE)
  FetchContent_MakeAvailable(xtensor-io)
endif()
