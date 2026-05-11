#ifndef MPI_CORE_H
#define MPI_CORE_H

#include <mpi.h>
#include <string>

/**
 * @class MPICore
 * @brief Static utility class for initializing and managing the MPI environment.
 */
class MPICore {
public:
    /**
     * @struct ProcessInfo
     * @brief Encapsulates standard information about the current MPI process.
     */
    struct ProcessInfo {
        int rank;       ///< The rank (ID) of the current process
        int size;       ///< The total number of processes in the communicator
        MPI_Comm comm;  ///< The active communicator
    };

    /**
     * @brief Initializes the MPI environment.
     * @param argc Pointer to the number of command-line arguments.
     * @param argv Pointer to the command-line arguments array.
     * @return A ProcessInfo struct populated with global rank and size.
     */
    static ProcessInfo init(int argc, char** argv);

    /**
     * @brief Cleans up and shuts down the MPI environment.
     */
    static void finalize();

    /**
     * @brief Splits an existing communicator into logical subgroups.
     * @param original_comm The communicator to split.
     * @param color Processes with the same color are grouped into the same new communicator.
     * @param key Controls rank assignment within the new communicator.
     * @return The newly created MPI_Comm.
     */
    static MPI_Comm split_communicator(MPI_Comm original_comm, int color, int key);

    /**
     * @brief Creates a 2D logical Cartesian topology from a standard communicator.
     * @param original_comm The base communicator (e.g., MPI_COMM_WORLD).
     * @param rows The number of processes in the Y dimension.
     * @param cols The number of processes in the X dimension.
     * @return A Cartesian MPI_Comm with periodic boundaries enabled.
     */
    static MPI_Comm create_cartesian_topology(MPI_Comm original_comm, int rows, int cols);

    /**
     * @brief Retrieves the current wall-clock time for benchmarking.
     * @return A double precision timestamp.
     */
    static double start_timer();

    /**
     * @brief Calculates the elapsed time since a given start point.
     * @param start_time The timestamp recorded by start_timer().
     * @return The elapsed execution duration in seconds.
     */
    static double stop_timer(double start_time);
};

#endif