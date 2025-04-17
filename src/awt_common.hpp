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


/*******************************************************************************
 *                              Helper Functions                               *
 *******************************************************************************/

using Matrix = std::vector<std::vector<float>>;
constexpr float INF = 1e9f;


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
    {1.0f, 0.0f},

    // Edge-aware high-bias average
    {0.95f, 0.05f},
    {0.9f, 0.1f},
    {0.85f, 0.15f},
    {0.8f, 0.2f},

    // Edge-aware low-bias average
    {0.75f, 0.25f},
    {0.7f, 0.3f},
    {0.65f, 0.35f},
    {0.6f, 0.4f},
    {0.55f, 0.45f},

    // Avg of left and right neighbors
    {0.5f, 0.5f},

    // Edge-aware low-bias average
    {0.45f, 0.55f},
    {0.4f, 0.6f},
    {0.35f, 0.65f},
    {0.3f, 0.7f},
    {0.25f, 0.75f},

    // Edge-aware high-bias average
    {0.2f, 0.8f},
    {0.15f, 0.85f},
    {0.1f, 0.9f},
    {0.05f, 0.95f},

    // Only Right
    {0.0f, 1.0f},

    // Four-point predictors with variations
    // Gaussian-like weighting
    {0.25f, 0.25f, 0.25f, 0.25f},
    {0.225f, 0.275f, 0.275f, 0.225f},
    {0.2f, 0.3f, 0.3f, 0.2f},
    {0.175f, 0.325f, 0.325f, 0.175f},
    {0.15f, 0.35f, 0.35f, 0.15f},
    {0.125f, 0.375f, 0.375f, 0.125f},
    {0.1f, 0.4f, 0.4f, 0.1f},
    {0.075f, 0.425f, 0.425f, 0.075f},
    {0.05f, 0.45f, 0.45f, 0.05f},
    {0.025f, 0.475f, 0.475f, 0.025f},

    // Asymmetric variations
    {0.1f, 0.3f, 0.4f, 0.2f},
    {0.2f, 0.4f, 0.3f, 0.1f},
    {0.15f, 0.25f, 0.4f, 0.2f},
    {0.2f, 0.4f, 0.25f, 0.15f},

    // Quadratic and high-order polynomials
    {-0.25f, 0.75f, 0.75f, -0.25f},
    {-0.2f, 0.7f, 0.7f, -0.2f},
    {-0.175f, 0.675f, 0.675f, -0.175f},
    {-0.15f, 0.65f, 0.65f, -0.15f},
    {-0.125f, 0.625f, 0.625f, -0.125f},
    {-0.1f, 0.6f, 0.6f, -0.1f},
    {-0.0875f, 0.5875f, 0.5875f, -0.0875f},
    {-0.075f, 0.575f, 0.575f, -0.075f},
    {-0.0625f, 0.5625f, 0.5625f, -0.0625f},
    {-0.05f, 0.55f, 0.55f, -0.05f},
    {-0.0375f, 0.5375f, 0.5375f, -0.0375f},
    {-0.025f, 0.525f, 0.525f, -0.025f},
    {-0.0125f, 0.5125f, 0.5125f, -0.0125f}};



/*******************************************************************************
 *                               AWT Updates                                   *
 *******************************************************************************/
    
const Matrix AWT_UPDATES = {
    // Only Left
    {0.5f, 0.0f},

    // Edge-aware high-bias average
    {0.475f, 0.025f},
    {0.45f, 0.05f},
    {0.425f, 0.075f},
    {0.4f, 0.1f},

    // Edge-aware low-bias average
    {0.375f, 0.125f},
    {0.35f, 0.15f},
    {0.325f, 0.175f},
    {0.3f, 0.2f},
    {0.275f, 0.225f},

    // Avg of left and right neighbors
    {0.25f, 0.25f},

    // Edge-aware low-bias average
    {0.225f, 0.275f},
    {0.2f, 0.3f},
    {0.175f, 0.325f},
    {0.15f, 0.35f},
    {0.125f, 0.375f},

    // Edge-aware high-bias average
    {0.1f, 0.4f},
    {0.075f, 0.425f},
    {0.05f, 0.45f},
    {0.025f, 0.475f},

    // Only Right
    {0.0f, 0.5f},

    // Four-point predictors with variations
    // Gaussian-like weighting
    {0.125f, 0.125f, 0.125f, 0.125f},
    {0.1125f, 0.1375f, 0.1375f, 0.1125f},
    {0.1f, 0.15f, 0.15f, 0.1f},
    {0.0875f, 0.1625f, 0.1625f, 0.0875f},
    {0.075f, 0.175f, 0.175f, 0.075f},
    {0.0625f, 0.1875f, 0.1875f, 0.0625f},
    {0.05f, 0.2f, 0.2f, 0.05f},
    {0.0375f, 0.2125f, 0.2125f, 0.0375f},
    {0.025f, 0.225f, 0.225f, 0.025f},
    {0.0125f, 0.2375f, 0.2375f, 0.0125f},

    // Asymmetric variations
    {0.05f, 0.15f, 0.2f, 0.1f},
    {0.1f, 0.2f, 0.15f, 0.05f},
    {0.075f, 0.125f, 0.2f, 0.1f},
    {0.1f, 0.2f, 0.125f, 0.075f},

    // Quadratic and high-order polynomials
    {-0.125f, 0.375f, 0.375f, -0.125f},
    {-0.1f, 0.35f, 0.35f, -0.1f},
    {-0.0875f, 0.3375f, 0.3375f, -0.0875f},
    {-0.075f, 0.325f, 0.325f, -0.075f},
    {-0.0625f, 0.3125f, 0.3125f, -0.0625f},
    {-0.05f, 0.3f, 0.3f, -0.05f},
    {-0.04375f, 0.29375f, 0.29375f, -0.04375f},
    {-0.0375f, 0.2875f, 0.2875f, -0.0375f},
    {-0.03125f, 0.28125f, 0.28125f, -0.03125f},
    {-0.025f, 0.275f, 0.275f, -0.025f},
    {-0.01875f, 0.26875f, 0.26875f, -0.01875f},
    {-0.0125f, 0.2625f, 0.2625f, -0.0125f},
    {-0.00625f, 0.25625f, 0.25625f, -0.00625f}};


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
        std::vector<float> row;
        std::istringstream iss(line);
        float val;

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