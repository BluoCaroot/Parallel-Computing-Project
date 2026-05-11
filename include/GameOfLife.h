#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include <vector>
#include <mpi.h>

/**
 * @class GameOfLife
 * @brief Simulates Conway's Game of Life over a 2D distributed grid using OpenMPI.
 *
 * @details This class handles the local segment of a global grid, utilizing a 2D
 * Cartesian topology to communicate with 8 Moore neighborhood.
 */
class GameOfLife {
private:
    /**
     * @struct ProcessNeighbors
     * @brief Stores the MPI ranks of the 8 surrounding neighbors in the Cartesian grid.
     */

    std::vector<int> local_grid;    ///< 1D vector representing the 2D local grid including ghost cells
    int local_rows;                 ///< Number of rows in the local grid (excluding ghost cells)
    int local_cols;                 ///< Number of columns in the local grid (excluding ghost cells)
    int global_rows;                ///< Total number of rows in the global grid
    int global_cols;                ///< Total number of columns in the global grid
    int neighbors[8];               ///< Ranks of neighbors (Up, Down, Left, Right, UL, UR, DL, DR)
    MPI_Comm cart_comm;             ///< Cartesian communicator for the 2D grid

    /**
     * @brief Computes the next state (alive/dead) for a single cell.
     * @param r The row index of the cell.
     * @param c The column index of the cell.
     * @return 1 if the cell will be alive in the next generation, 0 if dead.
     */
    int compute_next_state(int r, int c);

    /**
     * @brief Determines the MPI ranks for all 8 adjacent and diagonal neighbors.
     * @details Uses MPICore::get_cart_neighbors to populate the neighbors array.
     */
    void calculate_diagonal_neighbors();

    /**
     * @brief Demonstrates a classic MPI circular-wait deadlock scenario.
     * @details Uses MPICore::send_recv_blocking. If buffers fill, processes wait indefinitely.
     */
    void exchange_boundaries_deadlock();

    /**
     * @brief Correctly exchanges ghost cells using non-blocking communication.
     * @details Uses MPICore non-blocking wrappers to prevent deadlock.
     */
    void exchange_boundaries_nonblocking();

    /**
     * @brief Gathers the global grid to rank 0 and prints it to the terminal.
     * @param generation The current generation number.
     */
    void print_grid(int generation);

public:
    /**
     * @brief Constructor for the GameOfLife simulation.
     * @param local_r Number of rows assigned to this specific process.
     * @param local_c Number of columns assigned to this specific process.
     * @param global_r Total number of rows in the global grid.
     * @param global_c Total number of columns in the global grid.
     * @param comm The Cartesian MPI communicator to use.
     */
    GameOfLife(int local_r, int local_c, int global_r, int global_c, MPI_Comm comm);

    /**
     * @brief Executes the simulation for a given number of generations.
     * @param generations The number of iterations to run the Game of Life rules.
     */
    void run_simulation(int generations);


};

#endif