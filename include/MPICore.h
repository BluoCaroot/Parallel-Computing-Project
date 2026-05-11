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
     * @brief Retrieves the rank of the current process within a communicator.
     * @param comm The communicator.
     * @return The rank of the process.
     */
    static int get_rank(MPI_Comm comm);

    /**
     * @brief Retrieves the total number of processes within a communicator.
     * @param comm The communicator.
     * @return The number of processes.
     */
    static int get_size(MPI_Comm comm);

    /**
     * @brief Aborts the MPI execution environment.
     * @param comm The communicator.
     * @param error_code The error code to return.
     */
    static void abort(MPI_Comm comm, int error_code);

    /**
     * @brief Broadcasts a message from root to all other processes in the communicator.
     * @param buffer Pointer to the data to broadcast.
     * @param count Number of elements in the buffer.
     * @param type MPI datatype of the elements.
     * @param root Rank of the root process.
     * @param comm The communicator.
     */
    static void broadcast(void* buffer, int count, MPI_Datatype type, int root, MPI_Comm comm);

    /**
     * @brief Scatters data from root to all processes, allowing for uneven chunks.
     * @param send_buf Pointer to the global data (on root).
     * @param send_counts Array of counts for each process.
     * @param displacements Array of offsets for each process.
     * @param send_type MPI datatype of the sent elements.
     * @param recv_buf Pointer to the local receive buffer.
     * @param recv_count Number of elements to receive.
     * @param recv_type MPI datatype of the received elements.
     * @param root Rank of the root process.
     * @param comm The communicator.
     */
    static void scatterv(const void* send_buf, const int* send_counts, const int* displacements, MPI_Datatype send_type,
                         void* recv_buf, int recv_count, MPI_Datatype recv_type, int root, MPI_Comm comm);

    /**
     * @brief Gathers uneven chunks of data from all processes back to root.
     * @param send_buf Pointer to the local data.
     * @param send_count Number of elements to send.
     * @param send_type MPI datatype of the sent elements.
     * @param recv_buf Pointer to the global receive buffer (on root).
     * @param recv_counts Array of counts received from each process.
     * @param displacements Array of offsets for each process in the global buffer.
     * @param recv_type MPI datatype of the received elements.
     * @param root Rank of the root process.
     * @param comm The communicator.
     */
    static void gatherv(const void* send_buf, int send_count, MPI_Datatype send_type,
                        void* recv_buf, const int* recv_counts, const int* displacements, MPI_Datatype recv_type,
                        int root, MPI_Comm comm);

    /**
     * @brief Determines the MPI ranks for all 8 adjacent and diagonal neighbors in a 2D Cartesian topology.
     * @param comm The Cartesian communicator.
     * @param neighbors Array of 8 integers to store ranks (Up, Down, Left, Right, UL, UR, DL, DR).
     */
    static void get_cart_neighbors(MPI_Comm comm, int neighbors[8]);

    /**
     * @brief Performs a blocking send and receive.
     * @param send_data Pointer to the data to send.
     * @param send_count Number of elements to send.
     * @param dest Destination rank.
     * @param recv_data Pointer to the buffer for received data.
     * @param recv_count Number of elements to receive.
     * @param source Source rank.
     * @param comm The communicator.
     */
    static void send_recv_blocking(void* send_data, int send_count, int dest,
                                   void* recv_data, int recv_count, int source,
                                   MPI_Comm comm);

    /**
     * @brief Posts a non-blocking receive.
     * @param buf Pointer to the buffer for received data.
     * @param count Number of elements to receive.
     * @param type MPI datatype of the elements.
     * @param source Source rank.
     * @param tag Message tag.
     * @param comm The communicator.
     * @param request Pointer to the MPI_Request object.
     */
    static void irecv(void* buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm, MPI_Request* request);

    /**
     * @brief Posts a non-blocking send.
     * @param buf Pointer to the data to send.
     * @param count Number of elements to send.
     * @param type MPI datatype of the elements.
     * @param dest Destination rank.
     * @param tag Message tag.
     * @param comm The communicator.
     * @param request Pointer to the MPI_Request object.
     */
    static void isend(const void* buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm, MPI_Request* request);

    /**
     * @brief Waits for all non-blocking operations to complete.
     * @param count Number of requests.
     * @param requests Array of MPI_Request objects.
     */
    static void wait_all(int count, MPI_Request* requests);

    /**
     * @brief Creates and commits a vector MPI datatype.
     * @param count Number of blocks.
     * @param blocklength Number of elements in each block.
     * @param stride Spacing between start of each block.
     * @param oldtype Old MPI datatype.
     * @return The newly created and committed MPI_Datatype.
     */
    static MPI_Datatype create_vector_type(int count, int blocklength, int stride, MPI_Datatype oldtype);

    /**
     * @brief Frees an MPI datatype.
     * @param type Pointer to the MPI_Datatype to free.
     */
    static void free_type(MPI_Datatype* type);

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