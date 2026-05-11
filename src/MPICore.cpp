#include "MPICore.h"


MPICore::ProcessInfo MPICore::init(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    ProcessInfo info{};
    MPI_Comm_rank(MPI_COMM_WORLD, &info.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &info.size);
    info.comm = MPI_COMM_WORLD;
    return info;
}

void MPICore::finalize() {
    MPI_Finalize();
}

MPI_Comm MPICore::split_communicator(MPI_Comm original_comm, int color, int key) {
    MPI_Comm new_world;
    MPI_Comm_split(original_comm, color, key, &new_world);
    return new_world;
}

MPI_Comm MPICore::create_cartesian_topology(MPI_Comm original_comm, int rows, int cols) {
    MPI_Comm cart;
    int dims[]{rows, cols};
    int periods[]{1, 1};
    MPI_Cart_create(original_comm,
                    2,
                    dims,
                    periods,
                    true,
                    &cart);
    return cart;
}

double MPICore::start_timer() {
    return MPI_Wtime();
}

double MPICore::stop_timer(double start_time) {
    return MPI_Wtime() - start_time;
}