# Deadlock Exploration & Resolution

In distributed computing, deadlocks are often silent killers. This project provides a controlled environment to study and resolve one of the most common MPI pitfalls: **The Circular Wait**.

## 1. The Deadlock Scenario: "The Synchronous Squeeze"

In the Game of Life simulation, each process must exchange boundary data with its neighbors. A naive implementation might look like this:

```cpp
// PSEUDOCODE: DANGEROUS
for (neighbor : neighbors) {
    MPI_Send(my_boundary, dest);    // Blocking Send
    MPI_Recv(their_boundary, dest); // Blocking Receive
}
```

### Why this fails:
1.  **Circular Dependency**: In a 2D grid, Process A is sending to Process B, while Process B is sending to Process A.
2.  **The Buffer Limit**: MPI implementations typically buffer small messages. However, once a message exceeds the system's internal buffer size (or if the system runs out of buffer space), `MPI_Send` **blocks** until a matching `MPI_Recv` is called.
3.  **The Hang**: If Process A and Process B both call `MPI_Send` at the same time, and both messages require the other side to be at the `MPI_Recv` call to clear the buffer, they will wait for each other forever. Neither will ever reach the `MPI_Recv` line.

## 2. The Solution: Non-Blocking Orchestration

The engine resolves this using **Asynchronous Communication** in `GameOfLife::exchange_boundaries_nonblocking()`.

### The Strategy:
1.  **Post All Receives First**: We use `MPI_Irecv` (Immediate Receive) for all 8 neighbors. This tells the MPI controller, "I am ready to receive data whenever it arrives; put it in this buffer."
2.  **Post All Sends**: We use `MPI_Isend` (Immediate Send). This initiates the data transfer and returns control to the program immediately, regardless of whether the receiver has started yet.
3.  **Synchronization Barrier**: We call `MPI_Waitall`. This ensures that the program does not proceed to the calculation phase until all 16 operations (8 sends + 8 receives) are verified as complete.

### Why this works:
By decoupling the "Intent to Send/Receive" from the "Completion of Data Transfer," we break the circular wait. The order in which processes start no longer matters because every process has signaled its readiness to both send and receive simultaneously.

## 3. Code Implementation
The fix is implemented in `src/GameOfLife.cpp` using a `MPI_Request` array to track the status of all concurrent transfers, ensuring maximum throughput and zero deadlock risk.
