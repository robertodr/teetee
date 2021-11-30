include(FetchContent)

# declare Catch2
FetchContent_Declare(Catch2
  QUIET
  URL
    https://github.com/catchorg/Catch2/archive/refs/tags/v2.13.7.zip
  )

# make available
FetchContent_MakeAvailable(Catch2)

# testing binary
add_executable(unit_tests
  unit_tests.cpp
  test_tteigen.cpp
  )

target_link_libraries(unit_tests
  PRIVATE
    Catch2::Catch2
    teetee
  )

# automatic discovery of unit tests
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/contrib)
include(Catch)
catch_discover_tests(unit_tests
  PROPERTIES
    LABELS "unit"
    EXTRA_ARGS "--success"
  )
