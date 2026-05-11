#ifndef MATRIX_MULTIPLIER_H
#define MATRIX_MULTIPLIER_H

#include <vector>
#include <string>
#include <mpi.h>

/**
 * @class MatrixMultiplier
 * @brief Performs distributed matrix multiplication using a logical ring topology.
 */
class MatrixMultiplier {
private:
    std::vector<int> local_A;   ///< The horizontal slice of Matrix A assigned to this process
    std::vector<int> local_B;   ///< The current block of Matrix B being processed
    std::vector<int> local_C;   ///< The resulting partial products computed by this process

    int local_rows;             ///< Number of rows in local_A
    int global_M;               ///< Total number of rows in Matrix A
    int global_K;               ///< Total number of columns in Matrix A / rows in Matrix B
    int global_cols;            ///< Total number of columns in the global matrices
    MPI_Comm comm;              ///< The MPI communicator used for ring shifting

public:
    /**
     * @brief Constructor for the MatrixMultiplier.
     * @param comm The MPI communicator coordinating the multiplication.
     */
    MatrixMultiplier(MPI_Comm comm);

    /**
     * @brief Main driver function to execute the full matrix multiplication pipeline.
     * @param matA_file Path to the input file for Matrix A.
     * @param matB_file Path to the input file for Matrix B.
     */
    void run_multiplication(const std::string& matA_file, const std::string& matB_file);

    /**
     * @brief Scatters Matrix A unevenly to all processes and broadcasts initial state.
     * @param global_A The complete Matrix A.
     * @param global_B The complete Matrix B.
     */
    void distribute_matrices(const std::vector<int>& global_A, const std::vector<int>& global_B);

    /**
     * @brief Executes the parallel computation using a logical Ring Communication pattern.
     * @details Shifts blocks of Matrix B to neighboring processes iteratively, computing partial sums.
     */
    void multiply_ring_algorithm();

    /**
     * @brief Collects the uneven chunks of resulting data from all processes back to root.
     * @param global_C Output vector where the final, complete matrix will be stored (on root).
     */
    void gather_results(std::vector<int>& global_C);
};

#endif