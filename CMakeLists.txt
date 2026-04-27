cmake_minimum_required(VERSION 4.0)

project(Assignment6)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(3rd/benchmark)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
set(CMAKE_CXX_STANDARD 20)

add_executable(benchMonteCarlo src/BenchMonteCarlo.cpp)

target_link_libraries(benchMonteCarlo
        PRIVATE
        benchmark::benchmark
)

add_executable(monteCarlo src/MonteCarlo.cpp)