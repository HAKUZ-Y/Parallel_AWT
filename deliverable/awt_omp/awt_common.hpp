#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <numeric>


/*******************************************************************************
 *                              Helper Functions                               *
 *******************************************************************************/

using Matrix = std::vector<std::vector<double>>;
constexpr double INF = 1e9f;

namespace fs = std::filesystem;

std::string extract_base_name(const std::string &filepath) {
    fs::path path_obj(filepath);
    return path_obj.stem().string();
}

// mirror the boundary for predictions
// TODO: optimize
inline int mirror(int index, int length) {
    if (index < 0) {
        return -index;
    }

    if (index >= length) {
        return 2 * length - index - 2;
    }

    return index;
}

std::vector<double> generate_predictor(int n, double start, double end) {
    std::vector<double> result;

    double step = (end - start) / (n - 1);
    for (int i = 0; i < n; ++i) {
        result.push_back(start + i * step);
    }

    double total = std::accumulate(result.begin(), result.end(), 0.0);
    for (auto &val : result) {
        val /= total;
    }
    return result;
}

std::vector<double> generate_update(int n, double start, double end) {
    std::vector<double> result;

    double step = (end - start) / (n - 1);
    for (int i = 0; i < n; ++i) {
        result.push_back(start + i * step);
    }

    double total = std::accumulate(result.begin(), result.end(), 0.0);
    for (auto &val : result) {
        val /= total * 2;
    }

    return result;
}


/*******************************************************************************
 *                               AWT Predictors                                *
 *******************************************************************************/

// Predictors
const Matrix AWT_PREDICTORS = {

    // Length 2
    {0.5, 0.5},
    {0.6, 0.4},
    {0.4, 0.6},
    {0.7, 0.3},

    // Length 4
    {0.25, 0.25, 0.25, 0.25},
    {0.4, 0.2, 0.2, 0.2},
    {0.3, 0.3, 0.2, 0.2},
    {0.2, 0.4, 0.2, 0.2},

    // Length 8
    generate_predictor(8, 0.1, 0.2),
    generate_predictor(8, 0.2, 0.1),
    generate_predictor(8, 0.1, 0.4),
    generate_predictor(8, 0.4, 0.1),

    // Length 16
    generate_predictor(16, 0.1, 0.2),
    generate_predictor(16, 0.2, 0.1),
    generate_predictor(16, 0.1, 0.4),
    generate_predictor(16, 0.4, 0.1),

    // Length 32
    generate_predictor(32, 0.1, 0.2),
    generate_predictor(32, 0.2, 0.1),
    generate_predictor(32, 0.1, 0.4),
    generate_predictor(32, 0.4, 0.1),

    // Length 64
    generate_predictor(64, 0.1, 0.2),
    generate_predictor(64, 0.2, 0.1),
    generate_predictor(64, 0.1, 0.4),
    generate_predictor(64, 0.4, 0.1),

    // Length 128
    generate_predictor(128, 0.1, 0.2),
    generate_predictor(128, 0.2, 0.1),
    generate_predictor(128, 0.1, 0.4),
    generate_predictor(128, 0.4, 0.1),

    // Length 256
    generate_predictor(256, 0.1, 0.2),

    // Length 512
    generate_predictor(512, 0.1, 0.2),

    // Length 1000
    generate_predictor(1000, 0.1, 0.2),
};

/*******************************************************************************
 *                               AWT Updates                                   *
 *******************************************************************************/

 const Matrix AWT_UPDATES = {

    // Length 2
    {0.25, 0.25},
    {0.3, 0.2},
    {0.2, 0.3},
    {0.35, 0.15},

    // Length 4
    {0.125, 0.125, 0.125, 0.125},
    {0.2, 0.1, 0.1, 0.1},
    {0.15, 0.15, 0.1, 0.1},
    {0.1, 0.2, 0.1, 0.1},

    // Length 8
    generate_update(8, 0.1, 0.2),
    generate_update(8, 0.2, 0.1),
    generate_update(8, 0.1, 0.4),
    generate_update(8, 0.4, 0.1),

    // Length 16
    generate_update(16, 0.1, 0.2),
    generate_update(16, 0.2, 0.1),
    generate_update(16, 0.1, 0.4),
    generate_update(16, 0.4, 0.1),

    // Length 32
    generate_update(32, 0.1, 0.2),
    generate_update(32, 0.2, 0.1),
    generate_update(32, 0.1, 0.4),
    generate_update(32, 0.4, 0.1),

    // Length 64
    generate_update(64, 0.1, 0.2),
    generate_update(64, 0.2, 0.1),
    generate_update(64, 0.1, 0.4),
    generate_update(64, 0.4, 0.1),

    // Length 128
    generate_update(128, 0.1, 0.2),
    generate_update(128, 0.2, 0.1),
    generate_update(128, 0.1, 0.4),
    generate_update(128, 0.4, 0.1),

    // Length 256
    generate_update(256, 0.1, 0.2),

    // Length 512
    generate_update(512, 0.1, 0.2),

    // Length 1000
    generate_update(1000, 0.1, 0.2),
};



/*******************************************************************************
 *                               Image Processing                              *
 *******************************************************************************/

// Load image
void load_image_from_file(const std::string &filename, Matrix &img) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Could not open file in load_image_from_file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    while (std::getline(infile, line)) {
        std::vector<double> row;
        std::istringstream iss(line);
        double val;

        while (iss >> val) {
            row.push_back(val);
        }

        if (!row.empty()) {
            img.push_back(row);
        }
    }

    // check square image
    if (img.size() != img[0].size()) {
        std::cerr << "Error: Invalid image size " << img.size() << "x" << img[0].size() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Load image " << filename << " with size "
              << img.size() << "x" << img[0].size() << std::endl;
}

// Save image
void save_image_to_file(const std::string &filename, const Matrix &img) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error: Could not open file in save_image_to_file" << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    for (const auto &row : img) {
        for (size_t i = 0; i < img.size(); ++i) {
            outfile << row[i];
            if (i < img.size() - 1) {
                outfile << " ";
            }
        }
        outfile << std::endl;
    }
}