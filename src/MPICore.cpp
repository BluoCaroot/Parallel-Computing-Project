#include "MPICore.h"
#include <thread>
#include <chrono>

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

int MPICore::get_rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int MPICore::get_size(MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

void MPICore::abort(MPI_Comm comm, int error_code) {
    MPI_Abort(comm, error_code);
}

void MPICore::broadcast(void* buffer, int count, MPI_Datatype type, int root, MPI_Comm comm) {
    MPI_Bcast(buffer, count, type, root, comm);
}

void MPICore::scatterv(const void* send_buf, const int* send_counts, const int* displacements, MPI_Datatype send_type,
                      void* recv_buf, int recv_count, MPI_Datatype recv_type, int root, MPI_Comm comm) {
    MPI_Scatterv(send_buf, send_counts, displacements, send_type, recv_buf, recv_count, recv_type, root, comm);
}

void MPICore::gatherv(const void* send_buf, int send_count, MPI_Datatype send_type,
                     void* recv_buf, const int* recv_counts, const int* displacements, MPI_Datatype recv_type,
                     int root, MPI_Comm comm) {
    MPI_Gatherv(send_buf, send_count, send_type, recv_buf, recv_counts, displacements, recv_type, root, comm);
}

void MPICore::  get_cart_neighbors(MPI_Comm comm, int neighbors[8]) {
    // 0: Up, 1: Down, 2: Left, 3: Right
    MPI_Cart_shift(comm, 0, 1, &neighbors[0], &neighbors[1]);
    MPI_Cart_shift(comm, 1, 1, &neighbors[2], &neighbors[3]);

    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    // 4: UL (y-1, x-1)
    int ul_coords[2] = {coords[0] - 1, coords[1] - 1};
    MPI_Cart_rank(comm, ul_coords, &neighbors[4]);

    // 5: UR (y-1, x+1)
    int ur_coords[2] = {coords[0] - 1, coords[1] + 1};
    MPI_Cart_rank(comm, ur_coords, &neighbors[5]);

    // 6: DL (y+1, x-1)
    int dl_coords[2] = {coords[0] + 1, coords[1] - 1};
    MPI_Cart_rank(comm, dl_coords, &neighbors[6]);

    // 7: DR (y+1, x+1)
    int dr_coords[2] = {coords[0] + 1, coords[1] + 1};
    MPI_Cart_rank(comm, dr_coords, &neighbors[7]);
}

void MPICore::send_recv_blocking(void* send_data, int send_count, int dest,
                                void* recv_data, int recv_count, int source,
                                MPI_Comm comm) {
    MPI_Send(send_data, send_count, MPI_INT, dest, 0, comm);
    MPI_Recv(recv_data, recv_count, MPI_INT, source, 0, comm, MPI_STATUS_IGNORE);
}

void MPICore::irecv(void* buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm, MPI_Request* request) {
    MPI_Irecv(buf, count, type, source, tag, comm, request);
}

void MPICore::isend(const void* buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm, MPI_Request* request) {
    MPI_Isend(buf, count, type, dest, tag, comm, request);
}

void MPICore::wait_all(int count, MPI_Request* requests) {
    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
}

MPI_Datatype MPICore::create_vector_type(int count, int blocklength, int stride, MPI_Datatype oldtype) {
    MPI_Datatype newtype;
    MPI_Type_vector(count, blocklength, stride, oldtype, &newtype);
    MPI_Type_commit(&newtype);
    return newtype;
}

void MPICore::free_type(MPI_Datatype* type) {
    MPI_Type_free(type);
}

void MPICore::create_dims(int nnodes, int ndims, int* dims) {
    MPI_Dims_create(nnodes, ndims, dims);
}

void MPICore::get_cart_coords(MPI_Comm comm, int rank, int maxdims, int* coords) {
    MPI_Cart_coords(comm, rank, maxdims, coords);
}

void MPICore::get_cart_dims(MPI_Comm comm, int maxdims, int* dims) {
    int periods[2];
    int coords[2];
    MPI_Cart_get(comm, maxdims, dims, periods, coords);
}

void MPICore::send(const void* buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm) {
    MPI_Send(buf, count, type, dest, tag, comm);
}

void MPICore::recv(void* buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm) {
    MPI_Recv(buf, count, type, source, tag, comm, MPI_STATUS_IGNORE);
}

void MPICore::sleep_ms(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

double MPICore::stop_timer(double start_time) {
    return MPI_Wtime() - start_time;
}