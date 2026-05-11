#include "MatrixMultiplier.h"
#include "MPICore.h"
#include "DataUtils.h"
#include <mpi.h>
#include <vector>
#include <iostream>

MatrixMultiplier::MatrixMultiplier(MPI_Comm comm) 
    : comm(comm), local_rows(0), global_M(0), global_K(0), global_cols(0) {}

void MatrixMultiplier::run_multiplication(const std::string& matA_file, const std::string& matB_file) {
    int rank = MPICore::get_rank(comm);

    std::vector<int> global_A, global_B, global_C;

    if (rank == 0) {
        int rA, cA, rB, cB;
        try {
            global_A = DataUtils::load_matrix_from_file(matA_file, rA, cA);
            global_B = DataUtils::load_matrix_from_file(matB_file, rB, cB);
            if (cA != rB) {
                std::cerr << "Error: Matrix dimensions mismatch for multiplication." << std::endl;
                MPICore::abort(comm, 1);
            }
            global_M = rA; global_K = cA; global_cols = cB;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            MPICore::abort(comm, 1);
        }
    }

    MPICore::broadcast(&global_M, 1, MPI_INT, 0, comm);
    MPICore::broadcast(&global_K, 1, MPI_INT, 0, comm);
    MPICore::broadcast(&global_cols, 1, MPI_INT, 0, comm);

    distribute_matrices(global_A, global_B);

    double start = MPICore::start_timer();
    multiply_ring_algorithm();
    double elapsed = MPICore::stop_timer(start);

    if (rank == 0) {
        std::cout << "Multiplication completed in " << elapsed << " seconds." << std::endl;
    }

    gather_results(global_C);

    if (rank == 0) {
        DataUtils::save_matrix_to_file("data/result.txt", global_C, global_M, global_cols);
    }
}
// todo: optimize row_plan and k_plan and recv_counts and disps
void MatrixMultiplier::distribute_matrices(const std::vector<int>& global_A, const std::vector<int>& global_B) {
    int rank = MPICore::get_rank(comm);
    int size = MPICore::get_size(comm);

    auto row_plan = DataUtils::calculate_distribution(global_M, size);
    auto k_plan = DataUtils::calculate_distribution(global_K, size);

    local_rows = row_plan.counts[rank];
    local_A.resize(local_rows * global_K);

    std::vector<int> send_counts(size), disps(size);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            send_counts[i] = row_plan.counts[i] * global_K;
            disps[i] = row_plan.displacements[i] * global_K;
        }
    }
    MPICore::scatterv(global_A.data(), send_counts.data(), disps.data(), MPI_INT,
                      local_A.data(), (int)local_A.size(), MPI_INT, 0, comm);

    local_B.resize(k_plan.counts[rank] * global_cols);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            send_counts[i] = k_plan.counts[i] * global_cols;
            disps[i] = k_plan.displacements[i] * global_cols;
        }
    }
    MPICore::scatterv(global_B.data(), send_counts.data(), disps.data(), MPI_INT,
                      local_B.data(), (int)local_B.size(), MPI_INT, 0, comm);
}
//todo: optmize k_plan
void MatrixMultiplier::multiply_ring_algorithm() {
    int rank = MPICore::get_rank(comm);
    int size = MPICore::get_size(comm);

    auto k_plan = DataUtils::calculate_distribution(global_K, size);
    local_C.assign(local_rows * global_cols, 0);

    std::vector<int> current_B = local_B;
    int left_neighbor = (rank - 1 + size) % size;
    int right_neighbor = (rank + 1) % size;

    for (int step = 0; step < size; ++step) {
        int b_rank = (rank - step + size) % size;
        int K_j = k_plan.counts[b_rank];
        int K_offset = k_plan.displacements[b_rank];

        // Multiplication: local_C += A_{i, b_rank} * B_{b_rank}
        if (local_rows > 0 && K_j > 0) {
            for (int r = 0; r < local_rows; ++r) {
                int r_off = r * global_K + K_offset;
                int c_off = r * global_cols;
                for (int k = 0; k < K_j; ++k) {
                    int a_val = local_A[r_off + k];
                    int bk_off = k * global_cols;
                    for (int c = 0; c < global_cols; ++c) {
                        local_C[c_off + c] += a_val * current_B[bk_off + c];
                    }
                }
            }
        }

        if (step < size - 1) {
            int next_b_rank = (rank - (step + 1) + size) % size;
            std::vector<int> next_B(k_plan.counts[next_b_rank] * global_cols);
            
            MPI_Request reqs[2];
            // Use current_B.data() safely even if empty
            void* send_ptr = current_B.empty() ? nullptr : current_B.data();
            void* recv_ptr = next_B.empty() ? nullptr : next_B.data();

            MPICore::isend(send_ptr, (int)current_B.size(), MPI_INT, right_neighbor, 0, comm, &reqs[0]);
            MPICore::irecv(recv_ptr, (int)next_B.size(), MPI_INT, left_neighbor, 0, comm, &reqs[1]);
            MPICore::wait_all(2, reqs);
            
            current_B = std::move(next_B);
        }
    }
}
// todo: optimize row_plan and recv_counts and disps
void MatrixMultiplier::gather_results(std::vector<int>& global_C) {
    int rank = MPICore::get_rank(comm);
    int size = MPICore::get_size(comm);

    auto row_plan = DataUtils::calculate_distribution(global_M, size);

    std::vector<int> recv_counts(size), disps(size);
    for (int i = 0; i < size; ++i) {
        recv_counts[i] = row_plan.counts[i] * global_cols;
        disps[i] = row_plan.displacements[i] * global_cols;
    }

    if (rank == 0) {
        global_C.resize(global_M * global_cols);
    }

    MPICore::gatherv(local_C.data(), (int)local_C.size(), MPI_INT,
                     global_C.data(), recv_counts.data(), disps.data(), MPI_INT, 0, comm);
}
