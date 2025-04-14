#include "metrics.hpp"
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

using Matrix = std::vector<std::vector<float>>;
constexpr float INF = 1e9f;

// Predictors
// TODO: find more with some sources
// Assume odd lenght for now
const Matrix AWT_PREDICTORS = {
    // {0},
    // {0.5f, 0, 0.5f},
    // {-0.125f, 0, 0.25f, 0.75f, 0.25f, 0, -0.125f}};
    {0.5f, 0.5f},                    // avg of neighbors
    {-0.25f, 0.75f, 0.75f, -0.25f}}; // a more complex filter

const Matrix AWT_UPDATES = {           // to preserve structure around 2 * i
    {0.25f, 0.25f},                    // for the first predictor
    {-0.125f, 0.25f, 0.25f, -0.125f}}; // for the more complex predictor

/*******************************************************************************
 *                              Helper Functions                               *
 *******************************************************************************/

// Clamp the boundary for predictions
inline int
clamp(int x, int min, int max) {
    return std::max(min, std::min(max, x));
}

namespace fs = std::filesystem;

std::string extract_base_name(const std::string &filepath) {
    fs::path path_obj(filepath);
    return path_obj.stem().string();
}

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

/*******************************************************************************
 *                               AWT Transformation                            *
 *******************************************************************************/

// Apply 1D AWT, stores [approx | detail] in a vector
void awt_1d(std::vector<float> &data, std::vector<float> &coefs, std::vector<int> &predictors) {
    int n = data.size() / 2;
    coefs.resize(n);
    predictors.resize(n);

    // Predict step: predict odd indices from even indices
    for (int i = 0; i < n; ++i) {
        int odd_idx = 2 * i + 1;
        float min_err = INF;
        float min_coef = 0.0f;
        int pred_index = 0;

        // find the best predictor adaptively
        // TODO: optimize
        for (size_t p = 0; p < AWT_PREDICTORS.size(); ++p) {
            const auto &filter = AWT_PREDICTORS[p];
            int len = filter.size();
            int start_index = odd_idx - len / 2;
            float pred_val = 0.0f;

            // apply each predictor filter and compute prediction error
            for (int j = 0; j < len; ++j) {
                int idx = clamp(start_index + j, 0, data.size() - 1);
                if (idx % 2 == 0) {
                    pred_val += filter[j] * data[idx];
                }
            }

            float err = std::abs(data[odd_idx] - pred_val);
            if (err < min_err) {
                min_err = err;
                min_coef = data[odd_idx] - pred_val;
                pred_index = p;
            }
        }

        // original value - prediction value
        // reverse in the reconstruction
        coefs[i] = min_coef;
        predictors[i] = pred_index;
    }

    // Update step (assumes symmetric filters)
    // adaptive,
    std::vector<float> result(data.size()); // buffer

    for (int i = 0; i < n; ++i) {
        result[i] = data[2 * i];

        const auto &update_filter = AWT_UPDATES[predictors[i]];
        int update_filter_len = update_filter.size();
        int start = i - update_filter_len / 2;

        float update_sum = 0.0f;

        for (int j = 0; j < update_filter_len; ++j) {
            int idx = clamp(start + j, 0, n - 1); // TODO: padding for edges
            update_sum += update_filter[j] * coefs[idx];
        }

        result[i] += update_sum;  // update approx coefficient
        result[i + n] = coefs[i]; // detail in the second half
    }
    data = result;
}
/*******************************************************************************
 *                               AWT Reconstruction                            *
 *******************************************************************************/

// Reconstruct 1D AWT
void reconstruct_awt_1d(std::vector<float> &data, const std::vector<int> &predictors) {
    int n = data.size() / 2;
    std::vector<float> approx(n), coefs(n);
    for (int i = 0; i < n; ++i) { //[approx | detail]
        approx[i] = data[i];
        coefs[i] = data[i + n];
    }

    // Undo update step
    for (int i = 0; i < n; ++i) {
        // get the predictor used
        int pred_index = predictors[i];

        // undo the updates, using predictor based update filter
        const auto &update_filter = AWT_UPDATES[pred_index];
        int update_filter_len = update_filter.size();
        int start = i - update_filter_len / 2;
        float update_sum = 0.0f;

        for (int j = 0; j < update_filter_len; ++j) {
            int idx = clamp(start + j, 0, n - 1);
            update_sum += update_filter[j] * coefs[idx];
        }
        approx[i] -= update_sum;
    }

    // Undo predict step
    std::vector<float> reconstructed(2 * n);
    for (int i = 0; i < n; ++i) {
        int even_idx = 2 * i;
        int odd_idx = 2 * i + 1;

        reconstructed[even_idx] = approx[i];

        // get the predictor used
        int pred_index = predictors[i];

        const auto &predict_filter = AWT_PREDICTORS[pred_index];
        int predict_filter_len = predict_filter.size();
        int start = odd_idx - predict_filter_len / 2;
        float pred_val = 0.0f;

        for (int j = 0; j < predict_filter_len; ++j) {
            int idx = clamp(start + j, 0, 2 * n - 1);
            if (idx % 2 == 0) {
                int even_i = idx / 2;
                pred_val += predict_filter[j] * approx[even_i];
            }
        }
        reconstructed[odd_idx] = pred_val + coefs[i];
    }
    data = reconstructed;
}
void test(std::vector<float> signal) {
    std::vector<float> original = signal;
    std::vector<float> coefs;
    std::vector<int> predictors;

    std::cout << "Original:\n";
    for (float x : signal)
        std::cout << x << " ";
    std::cout << "\n";

    // Forward AWT
    awt_1d(signal, coefs, predictors);

    std::cout << "After AWT ([approx | detail]):\n";
    for (float x : signal)
        std::cout << x << " ";
    std::cout << "\n";

    // Reconstruct
    reconstruct_awt_1d(signal, predictors);

    std::cout << "Reconstructed:\n";
    for (float x : signal)
        std::cout << x << " ";
    std::cout << "\n";

    // MSE
    float mse = 0.0f;
    for (int i = 0; i < signal.size(); ++i) {
        float diff = signal[i] - original[i];
        mse += diff * diff;
    }
    mse /= signal.size();

    std::cout << "MSE: " << mse << "\n";
}

int main() {
    std::vector<float> signal = {1.1, 2.2, 3.3, 4.4, 5.5, 6.66, 7.77, 8.88, 9.99, 10};

    test(signal);
    return 0;
}