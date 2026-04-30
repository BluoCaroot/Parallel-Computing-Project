#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include <vector>
#include <mpi.h>

class GameOfLife {
private:
    struct ProcessNeighbors {
        int neighbors[8];
    };

    std::vector<int> local_grid;
    int local_rows;
    int local_cols;
    ProcessNeighbors neighbors;
    MPI_Comm cart_comm;

    int compute_next_state(int r, int c);

    void calculate_diagonal_neighbors();

public:
    GameOfLife(int local_r, int local_c, MPI_Comm comm);

    void run_simulation(int generations);

    void exchange_boundaries_deadlock();

    void exchange_boundaries_nonblocking();
};

#endif