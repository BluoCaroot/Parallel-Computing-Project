cmake_minimum_required(VERSION 3.28)
project(parallelProject)

set(CMAKE_CXX_STANDARD 17)

set(MPI_INCLUDE "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_LIB "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")

include_directories(${MPI_INCLUDE})
add_executable(parallelProject
        src/main.cpp
        src/MPICore.cpp
        src/DataUtils.cpp
        src/GameOfLife.cpp
        src/MatrixMultiplier.cpp)
target_link_libraries(parallelProject ${MPI_LIB})
target_include_directories(parallelProject PRIVATE include)
