#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

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

/*******************************************************************************
 *                               AWT Predictors                                *
 *******************************************************************************/

// Predictors
// TODO: find more with some sources
// Must be even length
const Matrix AWT_PREDICTORS = {

    // Only Left
    {1.0, 0.0},

    // Edge-aware high-bias average
    {0.95, 0.05},
    {0.9, 0.1},
    {0.85, 0.15},
    {0.8, 0.2},

    // Edge-aware low-bias average
    {0.75, 0.25},
    {0.7, 0.3},
    {0.65, 0.35},
    {0.6, 0.4},
    {0.55, 0.45},

    // Avg of left and right neighbors
    {0.5, 0.5},

    // Edge-aware low-bias average
    {0.45, 0.55},
    {0.4, 0.6},
    {0.35, 0.65},
    {0.3, 0.7},
    {0.25, 0.75},

    // Edge-aware high-bias average
    {0.2, 0.8},
    {0.15, 0.85},
    {0.1, 0.9},
    {0.05, 0.95},

    // Only Right
    {0.0, 1.0},

    // Four-point predictors with variations
    // Gaussian-like weighting
    {0.25, 0.25, 0.25, 0.25},
    {0.225, 0.275, 0.275, 0.225},
    {0.2, 0.3, 0.3, 0.2},
    {0.175, 0.325, 0.325, 0.175},
    {0.15, 0.35, 0.35, 0.15},
    {0.125, 0.375, 0.375, 0.125},
    {0.1, 0.4, 0.4, 0.1},
    {0.075, 0.425, 0.425, 0.075},
    {0.05, 0.45, 0.45, 0.05},
    {0.025, 0.475, 0.475, 0.025},

    // Asymmetric variations
    {0.1, 0.3, 0.4, 0.2},
    {0.2, 0.4, 0.3, 0.1},
    {0.15, 0.25, 0.4, 0.2},
    {0.2, 0.4, 0.25, 0.15},

    // Quadratic and high-order polynomials
    {-0.25, 0.75, 0.75, -0.25},
    {-0.2, 0.7, 0.7, -0.2},
    {-0.175, 0.675, 0.675, -0.175},
    {-0.15, 0.65, 0.65, -0.15},
    {-0.125, 0.625, 0.625, -0.125},
    {-0.1, 0.6, 0.6, -0.1},
    {-0.0875, 0.5875, 0.5875, -0.0875},
    {-0.075, 0.575, 0.575, -0.075},
    {-0.0625, 0.5625, 0.5625, -0.0625},
    {-0.05, 0.55, 0.55, -0.05},
    {-0.0375, 0.5375, 0.5375, -0.0375},
    {-0.025, 0.525, 0.525, -0.025},
    {-0.0125, 0.5125, 0.5125, -0.0125}};

/*******************************************************************************
 *                               AWT Updates                                   *
 *******************************************************************************/

const Matrix AWT_UPDATES = {
    // Only Left
    {0.5, 0.0},

    // Edge-aware high-bias average
    {0.475, 0.025},
    {0.45, 0.05},
    {0.425, 0.075},
    {0.4, 0.1},

    // Edge-aware low-bias average
    {0.375, 0.125},
    {0.35, 0.15},
    {0.325, 0.175},
    {0.3, 0.2},
    {0.275, 0.225},

    // Avg of left and right neighbors
    {0.25, 0.25},

    // Edge-aware low-bias average
    {0.225, 0.275},
    {0.2, 0.3},
    {0.175, 0.325},
    {0.15, 0.35},
    {0.125, 0.375},

    // Edge-aware high-bias average
    {0.1, 0.4},
    {0.075, 0.425},
    {0.05, 0.45},
    {0.025, 0.475},

    // Only Right
    {0.0, 0.5},

    // Four-point predictors with variations
    // Gaussian-like weighting
    {0.125, 0.125, 0.125, 0.125},
    {0.1125, 0.1375, 0.1375, 0.1125},
    {0.1, 0.15, 0.15, 0.1},
    {0.0875, 0.1625, 0.1625, 0.0875},
    {0.075, 0.175, 0.175, 0.075},
    {0.0625, 0.1875, 0.1875, 0.0625},
    {0.05, 0.2, 0.2, 0.05},
    {0.0375, 0.2125, 0.2125, 0.0375},
    {0.025, 0.225, 0.225, 0.025},
    {0.0125, 0.2375, 0.2375, 0.0125},

    // Asymmetric variations
    {0.05, 0.15, 0.2, 0.1},
    {0.1, 0.2, 0.15, 0.05},
    {0.075, 0.125, 0.2, 0.1},
    {0.1, 0.2, 0.125, 0.075},

    // Quadratic and high-order polynomials
    {-0.125, 0.375, 0.375, -0.125},
    {-0.1, 0.35, 0.35, -0.1},
    {-0.0875, 0.3375, 0.3375, -0.0875},
    {-0.075, 0.325, 0.325, -0.075},
    {-0.0625, 0.3125, 0.3125, -0.0625},
    {-0.05, 0.3, 0.3, -0.05},
    {-0.04375, 0.29375, 0.29375, -0.04375},
    {-0.0375, 0.2875, 0.2875, -0.0375},
    {-0.03125, 0.28125, 0.28125, -0.03125},
    {-0.025, 0.275, 0.275, -0.025},
    {-0.01875, 0.26875, 0.26875, -0.01875},
    {-0.0125, 0.2625, 0.2625, -0.0125},
    {-0.00625, 0.25625, 0.25625, -0.00625}};


/*******************************************************************************
 *                               Image Processing                              *
 *******************************************************************************/

// Load image
void load_image_from_file(const std::string &filename, Matrix &img) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Could not open file in load_image_from_file " << filename << std::endl;
        MPI_Finalize();
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
        MPI_Finalize();
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
        MPI_Finalize();
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