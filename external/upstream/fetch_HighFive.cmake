set(HighFive_pinned "2.3.1")

find_package(HighFive ${HighFive_pinned} CONFIG QUIET
  NO_CMAKE_PATH
  NO_CMAKE_PACKAGE_REGISTRY
  )

if(TARGET HighFive)
  message(STATUS "Using HighFive: ${HighFive_INCLUDE_DIR} (version ${HighFive_VERSION})")
else()
  message(STATUS "Suitable HighFive could not be located. Fetching!")
  include(FetchContent)

  FetchContent_Declare(HighFive
    QUIET
    URL
      https://github.com/BlueBrain/HighFive/archive/v${HighFive_pinned}.zip
    )

  set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
  set(HIGHFIVE_PARALLEL_HDF5 OFF CACHE BOOL "" FORCE)
  set(HIGHFIVE_USE_INSTALL_DEPS ON CACHE BOOL "" FORCE)
  set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" FORCE)
  set(HIGHFIVE_USE_XTENSOR ON CACHE BOOL "" FORCE)
  set(HIGHFIVE_USE_EIGEN ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(HighFive)
endif()
