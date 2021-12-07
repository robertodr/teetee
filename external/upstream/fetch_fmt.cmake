find_package(fmt 8.0 CONFIG QUIET
  NO_CMAKE_PATH
  NO_CMAKE_PACKAGE_REGISTRY
  )

if(TARGET fmt::fmt)
  message(STATUS "Found fmt: ${fmt_INCLUDE_DIR} (found version ${fmt_VERSION})")
else()
  message(STATUS "Suitable fmt could not be located. Fetching!")
  include(FetchContent)

  FetchContent_Declare(fmt
    QUIET
    URL
      https://github.com/fmtlib/fmt/archive/8.0.1.zip
    )

  FetchContent_MakeAvailable(fmt)
endif()
