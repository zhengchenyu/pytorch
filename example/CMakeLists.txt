project(example)
set(CMAKE_CXX_STANDARD 17)

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
message("SRCS=${SRCS}")
message("CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message("Torch_SOURCE_DIR=${Torch_SOURCE_DIR}")

include_directories(.)
include_directories(../torch/include)

add_executable(example ${SRCS})
target_link_libraries(example pthread torch_cpu)