#ifndef MATRIX_MULTIPLIER_H
#define MATRIX_MULTIPLIER_H

#include <vector>
#include <string>
#include <mpi.h>

class MatrixMultiplier {
private:
    std::vector<int> local_A;
    std::vector<int> local_B;
    std::vector<int> local_C;

    int local_rows;
    int global_cols;
    MPI_Comm comm;

public:
    MatrixMultiplier(MPI_Comm comm);

    void run_multiplication(const std::string& matA_file, const std::string& matB_file);

    void distribute_matrices(const std::vector<int>& global_A, const std::vector<int>& global_B);

    void multiply_ring_algorithm();

    void gather_results(std::vector<int>& global_C);
};

#endif