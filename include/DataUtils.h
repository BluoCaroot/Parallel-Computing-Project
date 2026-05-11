#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <vector>
#include <string>

/**
 * @class DataUtils
 * @brief Static utility class for handling large datasets and calculating load distributions.
 */
class DataUtils {
public:
    /**
     * @struct DistributionPlan
     * @brief Holds the calculated arrays required for MPI_Scatterv and MPI_Gatherv.
     */
    struct DistributionPlan {
        std::vector<int> counts;        ///< Array specifying the number of elements to send/receive per process.
        std::vector<int> displacements; ///< Array specifying the starting index for each process's chunk.
    };

    /**
     * @brief Calculates a balanced distribution plan for uneven data sizes.
     * @param total_elements The total size of the global dataset.
     * @param num_procs The number of MPI processes dividing the data.
     * @return A DistributionPlan containing exact counts and displacements to avoid stragglers.
     */
    static DistributionPlan calculate_distribution(int total_elements, int num_procs);

    /**
     * @brief Loads a 2D matrix from a text file into a 1D vector.
     * @param filepath The path to the input file.
     * @param rows Reference to an int where the parsed number of rows will be stored.
     * @param cols Reference to an int where the parsed number of columns will be stored.
     * @return A 1D std::vector containing the matrix data.
     */
    static std::vector<int> load_matrix_from_file(const std::string& filepath, int& rows, int& cols);

    /**
     * @brief Saves a processed matrix back to a text file.
     * @param filepath The destination path for the output file.
     * @param data The 1D vector containing the global matrix data.
     * @param rows The total number of rows.
     * @param cols The total number of columns.
     */
    static void save_matrix_to_file(const std::string& filepath, const std::vector<int>& data, int rows, int cols);
};

#endif