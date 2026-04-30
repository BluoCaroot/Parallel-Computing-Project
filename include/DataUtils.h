#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <vector>
#include <string>

class DataUtils {
public:
    struct DistributionPlan {
        std::vector<int> counts;
        std::vector<int> displacements;
    };

    static DistributionPlan calculate_distribution(int total_elements, int num_procs);

    static std::vector<int> load_matrix_from_file(const std::string& filepath, int& rows, int& cols);
    static void save_matrix_to_file(const std::string& filepath, const std::vector<int>& data, int rows, int cols);
};

#endif