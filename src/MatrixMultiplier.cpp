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
void MatrixMultiplier::distribute_matrices(const std::vector<int>& global_A, const std::vector<int>& global_B) {
    int rank = MPICore::get_rank(comm);
    int size = MPICore::get_size(comm);

    auto my_row_info = DataUtils::get_local_distribution(global_M, size, rank);
    local_rows = my_row_info.count;
    local_A.resize(local_rows * global_K);

    auto my_k_info = DataUtils::get_local_distribution(global_K, size, rank);
    local_B.resize(my_k_info.count * global_cols);

    std::vector<int> send_counts_A, disps_A;
    std::vector<int> send_counts_B, disps_B;

    if (rank == 0) {
        send_counts_A.resize(size);
        disps_A.resize(size);
        send_counts_B.resize(size);
        disps_B.resize(size);

        for (int i = 0; i < size; ++i) {
            auto target_row_info = DataUtils::get_local_distribution(global_M, size, i);
            send_counts_A[i] = target_row_info.count * global_K;
            disps_A[i] = target_row_info.displacement * global_K;

            auto target_k_info = DataUtils::get_local_distribution(global_K, size, i);
            send_counts_B[i] = target_k_info.count * global_cols;
            disps_B[i] = target_k_info.displacement * global_cols;
        }
    }

    int* sc_A_ptr = rank == 0 ? send_counts_A.data() : nullptr;
    int* d_A_ptr  = rank == 0 ? disps_A.data() : nullptr;
    int* sc_B_ptr = rank == 0 ? send_counts_B.data() : nullptr;
    int* d_B_ptr  = rank == 0 ? disps_B.data() : nullptr;

    const int* gA_ptr = rank == 0 ? global_A.data() : nullptr;
    const int* gB_ptr = rank == 0 ? global_B.data() : nullptr;

    MPICore::scatterv(gA_ptr, sc_A_ptr, d_A_ptr, MPI_INT,
                      local_A.data(), (int)local_A.size(), MPI_INT, 0, comm);

    MPICore::scatterv(gB_ptr, sc_B_ptr, d_B_ptr, MPI_INT,
                      local_B.data(), (int)local_B.size(), MPI_INT, 0, comm);
}


void MatrixMultiplier::multiply_ring_algorithm() {
    int rank = MPICore::get_rank(comm);
    int size = MPICore::get_size(comm);

    local_C.assign(local_rows * global_cols, 0);

    std::vector<int> current_B = local_B;
    int left_neighbor = (rank - 1 + size) % size;
    int right_neighbor = (rank + 1) % size;

    for (int step = 0; step < size; ++step) {
        int b_rank = (rank - step + size) % size;

        auto current_b_info = DataUtils::get_local_distribution(global_K, size, b_rank);
        int K_j = current_b_info.count;
        int K_offset = current_b_info.displacement;

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

            auto next_b_info = DataUtils::get_local_distribution(global_K, size, next_b_rank);
            std::vector<int> next_B(next_b_info.count * global_cols);
            
            MPI_Request reqs[2];
            void* send_ptr = current_B.empty() ? nullptr : current_B.data();
            void* recv_ptr = next_B.empty() ? nullptr : next_B.data();

            MPICore::isend(send_ptr, (int)current_B.size(), MPI_INT, right_neighbor, 0, comm, &reqs[0]);
            MPICore::irecv(recv_ptr, (int)next_B.size(), MPI_INT, left_neighbor, 0, comm, &reqs[1]);
            MPICore::wait_all(2, reqs);
            
            current_B = std::move(next_B);
        }
    }
}
void MatrixMultiplier::gather_results(std::vector<int>& global_C) {
    int rank = MPICore::get_rank(comm);
    int size = MPICore::get_size(comm);

    std::vector<int> recv_counts, disps;

    if (rank == 0) {
        global_C.resize(global_M * global_cols);
        recv_counts.resize(size);
        disps.resize(size);

        for (int i = 0; i < size; ++i) {
            auto row_info = DataUtils::get_local_distribution(global_M, size, i);
            recv_counts[i] = row_info.count * global_cols;
            disps[i] = row_info.displacement * global_cols;
        }
    }

    int* rc_ptr = rank == 0 ? recv_counts.data() : nullptr;
    int* d_ptr  = rank == 0 ? disps.data() : nullptr;
    int* gC_ptr = rank == 0 ? global_C.data() : nullptr;

    MPICore::gatherv(local_C.data(), (int)local_C.size(), MPI_INT,
                     gC_ptr, rc_ptr, d_ptr, MPI_INT, 0, comm);
}
