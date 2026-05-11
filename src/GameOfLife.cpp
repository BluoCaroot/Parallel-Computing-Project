#include "GameOfLife.h"
#include "MPICore.h"
#include <random>
#include <ctime>
#include <iostream>

GameOfLife::GameOfLife(int local_r, int local_c, MPI_Comm comm)
    : local_rows(local_r), local_cols(local_c), cart_comm(comm) {
    
    local_grid.assign((local_rows + 2) * (local_cols + 2), 0);

    int rank = MPICore::get_rank(cart_comm);
    std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)) + rank);
    std::uniform_int_distribution dis(0, 1);

    for (int r = 1; r <= local_rows; ++r) {
        for (int c = 1; c <= local_cols; ++c) {
            local_grid[r * (local_cols + 2) + c] = dis(gen);
        }
    }

    calculate_diagonal_neighbors();
}

void GameOfLife::calculate_diagonal_neighbors() {
    MPICore::get_cart_neighbors(cart_comm, neighbors);
}

int GameOfLife::compute_next_state(int r, int c) {
    int alive_neighbors = 0;
    int stride = local_cols + 2;

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0) continue;
            alive_neighbors += local_grid[(r + i) * stride + (c + j)];
        }
    }

    int current_state = local_grid[r * stride + c];
    if (current_state == 1) {
        return (alive_neighbors == 2 || alive_neighbors == 3) ? 1 : 0;
    } else {
        return (alive_neighbors == 3) ? 1 : 0;
    }
}

void GameOfLife::exchange_boundaries_deadlock() {
    int stride = local_cols + 2;
    
    // Naive implementation: Sequential blocking Send/Recv for each neighbor
    // This will deadlock if the message size exceeds the MPI system buffer.
    for (int i = 0; i < 8; ++i) {
        int dest = neighbors[i];
        // This is a simplified version of the logic to demonstrate deadlock
        // In reality, each neighbor would need a different pointer and count.
        // We just send 1 int to trigger the wait behavior.
        int dummy_send = 1;
        int dummy_recv = 0;
        MPICore::send_recv_blocking(&dummy_send, 1, dest, &dummy_recv, 1, dest, cart_comm);
    }
}

void GameOfLife::exchange_boundaries_nonblocking() {
    int stride = local_cols + 2;
    MPI_Request requests[16];
    int req_count = 0;

    // groups all elements in same column into one container
    MPI_Datatype column_type = MPICore::create_vector_type(local_rows, 1, stride, MPI_INT);

    // Up (0) -> receive into top ghost row
    MPICore::irecv(&local_grid[0 * stride + 1], local_cols, MPI_INT, neighbors[0], 0, cart_comm, &requests[req_count++]);

    // Down (1) -> receive into bottom ghost row
    MPICore::irecv(&local_grid[(local_rows + 1) * stride + 1], local_cols, MPI_INT, neighbors[1], 1, cart_comm, &requests[req_count++]);

    // Left (2) -> receive into left ghost column
    MPICore::irecv(&local_grid[1 * stride + 0], 1, column_type, neighbors[2], 2, cart_comm, &requests[req_count++]);

    // Right (3) -> receive into right ghost column
    MPICore::irecv(&local_grid[1 * stride + (local_cols + 1)], 1, column_type, neighbors[3], 3, cart_comm, &requests[req_count++]);
    

    // Corners
    MPICore::irecv(&local_grid[0 * stride + 0], 1, MPI_INT, neighbors[4], 4, cart_comm, &requests[req_count++]); // UL
    MPICore::irecv(&local_grid[0 * stride + (local_cols + 1)], 1, MPI_INT, neighbors[5], 5, cart_comm, &requests[req_count++]); // UR
    MPICore::irecv(&local_grid[(local_rows + 1) * stride + 0], 1, MPI_INT, neighbors[6], 6, cart_comm, &requests[req_count++]); // DL
    MPICore::irecv(&local_grid[(local_rows + 1) * stride + (local_cols + 1)], 1, MPI_INT, neighbors[7], 7, cart_comm, &requests[req_count++]); // DR


    // Up (0) -> send from top inner row
    MPICore::isend(&local_grid[1 * stride + 1], local_cols, MPI_INT, neighbors[0], 1, cart_comm, &requests[req_count++]);

    // Down (1) -> send from bottom inner row
    MPICore::isend(&local_grid[local_rows * stride + 1], local_cols, MPI_INT, neighbors[1], 0, cart_comm, &requests[req_count++]);

    // Left (2) -> send from left inner column
    MPICore::isend(&local_grid[1 * stride + 1], 1, column_type, neighbors[2], 3, cart_comm, &requests[req_count++]);

    // Right (3) -> send from right inner column
    MPICore::isend(&local_grid[1 * stride + local_cols], 1, column_type, neighbors[3], 2, cart_comm, &requests[req_count++]);


    // Corners
    MPICore::isend(&local_grid[1 * stride + 1], 1, MPI_INT, neighbors[4], 7, cart_comm, &requests[req_count++]); // UL -> DR of neighbor
    MPICore::isend(&local_grid[1 * stride + local_cols], 1, MPI_INT, neighbors[5], 6, cart_comm, &requests[req_count++]); // UR -> DL of neighbor
    MPICore::isend(&local_grid[local_rows * stride + 1], 1, MPI_INT, neighbors[6], 5, cart_comm, &requests[req_count++]); // DL -> UR of neighbor
    MPICore::isend(&local_grid[local_rows * stride + local_cols], 1, MPI_INT, neighbors[7], 4, cart_comm, &requests[req_count++]); // DR -> UL of neighbor

    // wait for all processes to finish communication before continuing
    MPICore::wait_all(req_count, requests);

    MPICore::free_type(&column_type);
}

void GameOfLife::run_simulation(int generations) {
    int stride = local_cols + 2;
    std::vector<int> next_grid((local_rows + 2) * (local_cols + 2), 0);

    for (int g = 0; g < generations; ++g) {
        exchange_boundaries_nonblocking();

        for (int r = 1; r <= local_rows; ++r) {
            for (int c = 1; c <= local_cols; ++c) {
                next_grid[r * stride + c] = compute_next_state(r, c);
            }
        }
        local_grid = next_grid;
    }
}