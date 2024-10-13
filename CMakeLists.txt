cmake_minimum_required(VERSION 3.0)

project(MPI_GCN_CPU LANGUAGES C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c99 -O2 -Wno-unused-variable -Wno-unused-function -g -Werror")

find_package(MPI REQUIRED)

add_executable(MPI_GCN_CPU
        main.c
        src/basic.c
        src/fileio.c
        src/typedef.c
        NeuralNet/activationLayer.c
        NeuralNet/gcnLayer.c
        NeuralNet/neuralNet.c
        NeuralNet/lossFunctions.c
        Matrix/Matrix.c
        Matrix/sparseMat.c
        includes/dropoutLayer.h
        NeuralNet/dropoutLayer.c
        includes/masking.h
        NeuralNet/masking.c
)

target_link_libraries(MPI_GCN_CPU m)
target_link_libraries(MPI_GCN_CPU ${MPI_C_LIBRARIES})
target_include_directories(MPI_GCN_CPU PUBLIC ${MPI_C_INCLUDE_PATH})
