#include "DataUtils.h"
#include <fstream>
#include <stdexcept>

DataUtils::DistributionPlan DataUtils::calculate_distribution(int total_elements, int num_procs) {
    DistributionPlan plan;
    plan.counts.assign(num_procs, total_elements / num_procs);
    for (int i = 0; i < total_elements % num_procs; ++i)
        plan.counts[i]++;
    plan.displacements.resize(num_procs);
    for (int i = 1; i < num_procs; ++i)
        plan.displacements[i] = plan.displacements[i - 1] + plan.counts[i - 1];
    return plan;
}

DataUtils::LocalDistributionPlan DataUtils::get_local_distribution(int total_elements, int num_procs, int rank) {
    LocalDistributionPlan plan;
    int base = total_elements / num_procs;
    int remainder = total_elements % num_procs;
    plan.count = base + (rank < remainder ? 1 : 0);
    if (rank < remainder) {
        plan.displacement = rank * (base + 1);
    } else {
        plan.displacement = remainder * (base + 1) + (rank - remainder) * base;
    }
    return plan;
}

std::vector<int> DataUtils::load_matrix_from_file(const std::string& filepath, int& rows, int& cols) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    if (!(file >> rows >> cols)) {
        throw std::runtime_error("Failed to read dimensions from: " + filepath);
    }

    int total_elements = rows * cols;
    std::vector<int> data(total_elements);

    for (int i = 0; i < total_elements; ++i) {
        if (!(file >> data[i])) {
            throw std::runtime_error("Failed to read matrix data from: " + filepath);
        }
    }

    return data;
}

void DataUtils::save_matrix_to_file(const std::string& filepath, const std::vector<int>& data, int rows, int cols) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    file << rows << " " << cols << "\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << data[i * cols + j] << (j == cols - 1 ? "" : " ");
        }
        file << "\n";
    }
}