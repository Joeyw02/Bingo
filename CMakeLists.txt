cmake_minimum_required(VERSION 3.10)
project(MyProject VERSION 1.0 LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# 指定 include 目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 找到所有的源文件
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# 创建一个可执行文件
add_executable(MyExecutable ${SOURCES})

# 查找并链接 OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MyExecutable PRIVATE OpenMP::OpenMP_CXX)
endif()

# 查找并链接 CUDA 库
find_package(CUDA REQUIRED)

# 设置 CUDA 编译选项（可选）
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -O3")  # 根据你的 GPU 架构设置

# 指定 CUDA 目标属性
set_target_properties(MyExecutable PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
