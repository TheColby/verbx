cmake_minimum_required(VERSION 3.16)

project(verbx_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(VERBX_C_CORE_SOURCES
  src/audio.c
  src/algo_reverb.c
  src/render.c
  src/wav_io.c
  src/plugin_params.c
)

add_library(verbx_c_core STATIC ${VERBX_C_CORE_SOURCES})
target_include_directories(
  verbx_c_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(verbx_c_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NOT MSVC)
  target_link_libraries(verbx_c_core PUBLIC m)
endif()

add_executable(
  verbx-c
  src/main.c
  src/cli.c
)

target_link_libraries(verbx-c PRIVATE verbx_c_core)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(verbx-c PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)

if(BUILD_TESTING)
  add_executable(test_plugin_params tests/test_plugin_params.c)
  target_link_libraries(test_plugin_params PRIVATE verbx_c_core)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(test_plugin_params PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME plugin_params COMMAND test_plugin_params)
endif()

install(TARGETS verbx-c RUNTIME DESTINATION bin)
