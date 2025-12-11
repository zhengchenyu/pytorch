project(example)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
message("SRCS=${SRCS}")
message("CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message("Torch_SOURCE_DIR=${Torch_SOURCE_DIR}")

include_directories(.)
include_directories(../torch/include)
include_directories(../third_party/gloo)
include_directories(..)

add_executable(example ${SRCS}
        gloo_example.cpp
        process_group_example.cpp)
target_link_libraries(example pthread torch_cpu torch_cpu gloo)