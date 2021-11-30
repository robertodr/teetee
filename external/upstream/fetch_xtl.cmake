set(xtl_pinned "0.7.0")

find_package(xtl ${xtl_pinned} CONFIG QUIET
  NO_CMAKE_PATH
  NO_CMAKE_PACKAGE_REGISTRY
  )

if(TARGET xtl)
  message(STATUS "Found xtl: ${xtl_INCLUDE_DIR} (found version ${xtl_VERSION})")
else()
  message(STATUS "Suitable xtl could not be located. Fetching!")
  include(FetchContent)

  FetchContent_Declare(xtl
    QUIET
    URL
      https://github.com/QuantStack/xtl/archive/${xtl_pinned}.zip
    )

  set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(xtl)
endif()
