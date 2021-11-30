set(xsimd_pinned "8.0.3")

find_package(xsimd ${xsimd_pinned} CONFIG QUIET
  NO_CMAKE_PATH
  NO_CMAKE_PACKAGE_REGISTRY
  )

if(TARGET xsimd)
  message(STATUS "Found xsimd: ${xsimd_INCLUDE_DIR} (found version ${xsimd_VERSION})")
else()
  message(STATUS "Suitable xsimd could not be located. Fetching!")
  include(FetchContent)

  FetchContent_Declare(xsimd
    QUIET
    URL
      https://github.com/QuantStack/xsimd/archive/${xsimd_pinned}.zip
    )

  set(ENABLE_XTL_COMPLEX ON CACHE BOOL "" FORCE)
  set(xtl_DIR ${FETCHCONTENT_BASE_DIR}/xtl-build CACHE STRING "" FORCE)
  FetchContent_MakeAvailable(xsimd)
endif()
