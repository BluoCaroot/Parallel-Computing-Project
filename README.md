# Distributed Parallel Processing Engine

A high-performance C++ engine utilizing **OpenMPI** for distributed simulations and numerical computation. This project demonstrates advanced parallel patterns including 2D Cartesian decomposition, Ring communication algorithms, and deadlock-resilient synchronization.

## Overview

This engine implements two primary parallel workloads designed to showcase different distributed computing strategies:

1.  **Distributed Game of Life**: A 2D grid simulation using **Cartesian Topology**. It features a Moore neighborhood synchronization model with ghost-cell exchanges.
2.  **Distributed Matrix Multiplication**: A numerical implementation using a **Logical Ring Algorithm**, optimized for memory-constrained environments by shifting data blocks between processes.

## Technical Architecture

The project is structured to decouple MPI orchestration from domain logic:

-   **`MPICore`**: An abstraction layer for MPI primitives (Topologies, Non-blocking I/O, Timers).
-   **`GameOfLife`**: Implements 2D spatial decomposition.
-   **`MatrixMultiplier`**: Implements 1D row-based decomposition with ring shifting.
-   **`DataUtils`**: Provides load-balancing distribution plans and efficient I/O for large datasets.

## Documentation Deep-Dives

Detailed technical documentation is available in the `docs/` directory:

-   [**Architecture & Design**](docs/ARCHITECTURE.md): Detailed breakdown of 2D decomposition and communication strategies.
-   [**Deadlock Analysis**](docs/DEADLOCK_ANALYSIS.md): Exploration of the circular-wait deadlock scenario and the non-blocking resolution.
-   [**Performance Metrics**](docs/PERFORMANCE.md): Analysis of scaling, communication overhead, and efficiency.

## Build & Execution

### Prerequisites
-   OpenMPI (v4.0+)
-   C++11 Compatible Compiler (GCC/Clang)
-   CMake (v3.10+)

### Building
```bash
mkdir build && cd build
cmake ..
make
```

### Running
**Game of Life (2D Decomposition):**
```bash
mpirun -np 4 ./parallel_processor --algo gol --rows 1000 --cols 1000 --gen 100
```

**Matrix Multiplication (Ring Algorithm):**
```bash
mpirun -np 4 ./parallel_processor --algo matmult --fileA ../data/matA_large.txt --fileB ../data/matB_large.txt
```

---
*Developed as a demonstration of distributed systems engineering.*
