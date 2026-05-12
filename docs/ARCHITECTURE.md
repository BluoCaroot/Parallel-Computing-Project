# System Architecture & Communication

This document outlines the design principles and parallel strategies used in the Distributed Parallel Processing Engine.

## 1. Design Philosophy
The system is built on a **Modular Abstraction** model. By wrapping MPI calls within `MPICore`, the domain logic (Game of Life, Matrix Math) remains focused on the algorithm rather than the intricacies of message passing.

## 2. Communication Strategies

### A. 2D Cartesian Decomposition (Game of Life)
The global grid is treated as a 2D surface. Instead of simple row-wise slicing, we use a **Grid Partitioning** approach:
-   **Topology**: Uses `MPI_Cart_create` to map ranks to a $(P_y, P_x)$ grid.
-   **Halo Exchange**: Each process maintains a "ghost cell" border.
-   **Neighborhood**: Supports Moore neighborhood communication. Each process communicates with 8 neighbors (Cardinal + Ordinal directions).
-   **Efficiency**: 2D decomposition reduces the surface-to-volume ratio compared to 1D slicing, minimizing the total data sent as the number of processes increases.

### B. Logical Ring Algorithm (Matrix Multiplication)
To multiply $C = A \times B$ in a distributed environment:
-   **Decomposition**: Both Matrix $A$ and Matrix $B$ are sliced horizontally (row).
-   **The Ring**:  Processes are organized in a logical ring. Each process holds its fixed slice of $A$ and a rotating slice of $B$.
-   **Data Shifting**: After computation, the block of $B$ is shifted to the "Right" neighbor while receiving a new block from the "Left" neighbor.
-   **Benefit**: This algorithm is highly memory-efficient as no single process ever needs to hold the entire Matrix $B$ simultaneously during the computation phase.

## 3. Load Balancing
The engine employs a **Static Fair-Share** distribution strategy via `DataUtils::calculate_distribution`:
1.  **Base Allocation**: $Total / Size$.
2.  **Remainder Handling**: The first $Total \pmod{Size}$ processes receive one extra element.
3.  **Impact**: This ensures that even with non-divisible dimensions (e.g., 1000 rows on 3 processes), the workload difference is at most 1 unit, preventing significant synchronization bottlenecks (stragglers).
