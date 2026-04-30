# Advanced Parallel Grid Processing with MPI

This repository contains a C++ distributed processing system built with MPI (Message Passing Interface). The project demonstrates advanced parallel decomposition, inter-process communication, and scalability across 2D grid and matrix-based data structures.

## Features & Algorithms

This system implements two distinct parallel algorithms, falling into two categories (Grid/Spatial) and (Data/Computation), satisfying their requirements:

1. **Conway's Game of Life (Category A)**
   * **Data Distribution:** 2D Cartesian block decomposition using `MPI_Cart_create`.
   * **Communication:** Halo/Ghost cell exchange with neighboring processes.
   * **Techniques:** Explores both Blocking (Deadlock scenario) and Non-blocking (`MPI_Isend` / `MPI_Irecv`) communication.

2. **Matrix Multiplication (Category B)**
   * **Data Distribution:** 1D Row-wise decomposition capable of handling **uneven data sizes** (when matrix dimensions are not perfectly divisible by the number of processes).
   * **Communication:** Pipeline/Ring communication.
   * **Techniques:** Combines Collective communication (`MPI_Bcast`, `MPI_Scatterv`, `MPI_Gatherv`) with Point-to-Point logical ring shifting.

### Core MPI Mechanisms Used
* **Process Organization:** `MPI_Comm_split` is used to dynamically group processes based on the algorithm selected at runtime.
* **Large Data:** Utilizes dynamically allocated `std::vector` structures and handles loading/saving large datasets from disk.
* **Deadlock Handling:** Explicitly documents and solves a classic circular-wait deadlock.

---

##  Prerequisites & Building

To compile and run this project, you need an MPI implementation (like OpenMPI or MPICH) and a C++11 (or higher) compatible compiler.

### Prerequisites (Linux/macOS)
```bash
sudo apt-get install openmpi-bin libopenmpi-dev  # Ubuntu/Debian
brew install open-mpi                            # macOS
```

### Compilation
You can compile the project using the MPI C++ wrapper:
```bash
mpicxx -std=c++11 -O3 -Iinclude src/*.cpp -o parallel_processor
```

---

##  Running the System

The program allows you to select the algorithm at runtime via command-line arguments.

**Run Game of Life:**
```bash
# Runs with 4 processes, a 1000x1000 grid, for 100 generations
mpirun -np 4 ./parallel_processor --algo gol --rows 1000 --cols 1000 --gen 100
```

**Run Matrix Multiplication:**
```bash
# Runs with 4 processes using input files for Matrix A and Matrix B
mpirun -np 4 ./parallel_processor --algo matmult --fileA matA.txt --fileB matB.txt
```

---

##  Project Structure

```text
├── src/
│   ├── main.cpp                 # Entry point, argument parsing, runtime selection
│   ├── MPICore.cpp              # Environment initialization, timing, topology splitting
│   ├── DataUtils.cpp            # File I/O, uneven distribution plans (Scatterv/Gatherv)
│   ├── GameOfLife.cpp           # GoL logic, 2D topology, neighbor exchange
│   └── MatrixMultiplier.cpp     # MatMult logic, ring communication, block shifting
├── include/                     # Header files containing class definitions and prototypes
│   ├── MPICore.h                # Declarations for MPI init, timing, and communicators
│   ├── DataUtils.h              # Declarations for data distribution plans and file I/O
│   ├── GameOfLife.h             # Class definition for GoL, neighbor structs, and boundaries
│   └── MatrixMultiplier.h       # Class definition for matrix operations and ring communication
├── data/                        # Sample large matrices for testing
├── docs/                        # Demo screenshots and performance graphs
└── README.md                    # Project documentation
```

---

## Short Documentation

### 1. Design Overview
The system follows an Object-Oriented architecture. `MPICore` and `DataUtils` act as static utility classes managing the global state, timing, and heavy data I/O. The algorithms are encapsulated in instantiated classes (`GameOfLife` and `MatrixMultiplier`), which hold their local sub-grids/matrices and their specific communicators, preventing state leakage and keeping function signatures clean.

### 2. Communication Strategies
* **Game of Life:** Uses Non-Blocking Point-to-Point communication (`MPI_Isend` / `MPI_Irecv`). To accurately calculate the Game of Life rules across grid boundaries, each process performs an 8-way Halo/Ghost cell exchange with its adjacent and diagonal neighbors (Moore neighborhood) simultaneously to minimize idle time.
* **Matrix Multiplication:** Uses Collectives (`MPI_Scatterv` to distribute Matrix A, `MPI_Bcast` for initial state) and a **Ring Communication Pattern**. Processes shift their local blocks of Matrix B to `rank + 1` (and receive from `rank - 1`) in a loop to compute partial matrix products without broadcasting the entire matrix at once.

### 3. Deadlock Explanation & Solution
* **The Scenario:** In the Game of Life, if all processes attempt to send their right-side boundary using standard blocking `MPI_Send` at the same time, they must wait for the receiving process to call `MPI_Recv`.
* **Why it happens:** In a ring or periodic grid, process 0 sends to 1, 1 sends to 2, and process N sends to 0. If the grid is large enough to exceed MPI's internal eager-send buffers, every process blocks on `MPI_Send`, waiting for an `MPI_Recv` that will never be called. This is a classic "Circular Wait" deadlock.
* **The Solution:** The `GameOfLife::exchange_boundaries_nonblocking()` method fixes this. By posting all `MPI_Irecv` (receives) first, followed by `MPI_Isend` (sends), and then calling `MPI_Waitall`, the processes do not block each other, allowing the MPI runtime to resolve the data transfers asynchronously.

### 4. Performance Observations
* **Scalability:** As the number of processes (N) increases from 2 to 8, computation time decreases nearly linearly for Matrix Multiplication.
* **Communication Impact:** For Conway's Game of Life on very small grids (e.g., 50x50), adding more processes actually *slows down* the total execution time because the network communication overhead outweighs the computation time. For large grids (e.g., 5000x5000), the parallel speedup becomes highly visible.
* **Load Balancing:** `MPI_Scatterv` successfully prevented straggler processes by ensuring uneven remainders (e.g., 100 rows divided by 3 processes) were distributed fairly (34, 33, 33) rather than crashing or leaving one process idle.