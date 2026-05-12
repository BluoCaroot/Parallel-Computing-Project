#include "MPICore.h"
#include "MatrixMultiplier.h"
#include "GameOfLife.h"
#include "DataUtils.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    MPICore::ProcessInfo info = MPICore::init(argc, argv);

    std::string algo;
    int rows = 1000, cols = 1000, gen = 100;
    std::string fileA = "data/matA.txt", fileB = "data/matB.txt";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--algo" && i + 1 < argc) {
            algo = argv[++i];
        } else if (arg == "--rows" && i + 1 < argc) {
            rows = std::stoi(argv[++i]);
        } else if (arg == "--cols" && i + 1 < argc) {
            cols = std::stoi(argv[++i]);
        } else if (arg == "--gen" && i + 1 < argc) {
            gen = std::stoi(argv[++i]);
        } else if (arg == "--fileA" && i + 1 < argc) {
            fileA = argv[++i];
        } else if (arg == "--fileB" && i + 1 < argc) {
            fileB = argv[++i];
        }
    }

    if (algo == "gol") {
        int dims[2] = {0, 0};
        MPICore::create_dims(info.size, 2, dims);
        
        MPI_Comm cart_comm = MPICore::create_cartesian_topology(info.comm, dims[0], dims[1]);
        
        int coords[2];
        MPICore::get_cart_coords(cart_comm, info.rank, 2, coords);
        
        auto row_plan = DataUtils::calculate_distribution(rows, dims[0]);
        auto col_plan = DataUtils::calculate_distribution(cols, dims[1]);
        
        GameOfLife gol(row_plan.counts[coords[0]], col_plan.counts[coords[1]], rows, cols, cart_comm);
        
        double start = MPICore::start_timer();
        gol.run_simulation(gen);
        double elapsed = MPICore::stop_timer(start);
        
        if (info.rank == 0) {
            std::cout << "Game of Life simulation (" << rows << "x" << cols << ") for " 
                      << gen << " generations completed in " << elapsed << " seconds." << std::endl;
        }
    } else if (algo == "matmult") {
        MatrixMultiplier multiplier(info.comm);
        try {
            multiplier.run_multiplication(fileA, fileB);
        } catch (const std::exception& e) {
            if (info.rank == 0) {
                std::cerr << "Matrix Multiplication error: " << e.what() << std::endl;
            }
        }
    } else {
        if (info.rank == 0) {
            std::string s(argv[0]);
            std::string name = s.substr(s.find_last_of('\\') + 1);
            std::cout << "Usage:\n"
                      << "  Game of Life: mpirun -np <N> ./" << name << " --algo gol --rows <R> --cols <C> --gen <G>\n"
                      << "  Matrix Mult:  mpirun -np <N> ./" << name << " --algo matmult --fileA <A.txt> --fileB <B.txt>\n";
        }
    }

    MPICore::finalize();
    return 0;
}
