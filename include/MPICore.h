#ifndef MPI_CORE_H
#define MPI_CORE_H

#include <mpi.h>
#include <string>

class MPICore {
public:
    struct ProcessInfo {
        int rank;
        int size;
        MPI_Comm comm;
    };

    static ProcessInfo init(int argc, char** argv);
    static void finalize();

    static MPI_Comm split_communicator(MPI_Comm original_comm, int color, int key);
    static MPI_Comm create_cartesian_topology(MPI_Comm original_comm, int rows, int cols);

    static double start_timer();
    static double stop_timer(double start_time);
};

#endif