cmake_minimum_required(VERSION 3.10)
project(MyProject VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# include
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

add_executable(Bingo ${SOURCES})

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Bingo PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 -O3 --maxrregcount=128") 

set_target_properties(Bingo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
