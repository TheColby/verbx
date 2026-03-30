cmake_minimum_required(VERSION 3.16)

project(verbx_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(
  verbx-c
  src/main.c
  src/cli.c
)

target_include_directories(
  verbx-c
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(verbx-c PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS verbx-c RUNTIME DESTINATION bin)
