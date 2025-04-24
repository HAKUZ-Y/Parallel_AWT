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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using Matrix = std::vector<std::vector<float>>;
constexpr float INF = 1e9f;

// Predictors
// TODO: find more with some sources
// Assume odd length for now
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

void apply_threshold(Matrix &coeffs, float threshold) {
    for (auto &row : coeffs) {
        for (auto &val : row) {
            if (std::abs(val) < threshold) {
                val = 0.0f;
            }
        }
    }
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
void awt_1d(std::vector<float> &data, std::vector<float> &coefs, std::vector<float> &predictors) {
    int n = data.size() / 2;
    coefs.resize(n);
    predictors.resize(n);

// Predict step: predict odd indices from even indices
#pragma omp parallel for default(shared) schedule(static)
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

#pragma omp parallel for default(shared) schedule(static)
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

/**
 * apply 1D transformation to rows then cols, resulting in LL, HL, LH, HH
 * row-wise lifting ([approx|detail]) => LL, HL
 * col-wise lifting (also [approx|detail]) => LL, LH
 * diagonal lifting => HH, HL
 */
void awt_2d(Matrix &img,
            Matrix &h_coefs,
            Matrix &v_coefs,
            Matrix &d_coefs,
            Matrix &row_pred_map,
            Matrix &col_pred_map,
            Matrix &diag_pred_map) {
    int rows = img.size();
    int cols = img[0].size();

// row-wise transform
#pragma omp parallel for default(shared) schedule(static)
    for (int r = 0; r < rows; ++r) {
        std::vector<float> row = img[r];
        std::vector<float> row_coef;
        std::vector<float> row_predictors;

        awt_1d(row, row_coef, row_predictors);

        for (int c = 0; c < cols / 2; ++c) {
            img[r][c] = row[c];                // LL
            h_coefs[r][c] = row[c + cols / 2]; // HL approx
            row_pred_map[r][c] = row_predictors[c];
        }
    }

// col-wise transform
#pragma omp parallel for default(shared) schedule(static)
    for (int c = 0; c < cols / 2; ++c) {
        std::vector<float> col(rows);
        std::vector<float> col_coef;
        std::vector<float> col_predictors;

        for (int r = 0; r < rows; ++r) {
            col[r] = img[r][c]; // LL column
        }
        awt_1d(col, col_coef, col_predictors);

        for (int r = 0; r < rows / 2; ++r) {
            img[r][c] = col[r];                // final LL
            v_coefs[r][c] = col[r + rows / 2]; // LH detail
            col_pred_map[r][c] = col_predictors[r];
        }
    }

// get diagonal coefficients
#pragma omp parallel for default(shared) schedule(static)
    for (int c = 0; c < cols / 2; ++c) {
        std::vector<float> diag(rows);
        std::vector<float> diag_coefs;
        std::vector<float> diag_predictors;

        for (int r = 0; r < rows; ++r) {
            diag[r] = h_coefs[r][c]; // HL column
        }
        awt_1d(diag, diag_coefs, diag_predictors);

        for (int r = 0; r < rows / 2; ++r) {
            h_coefs[r][c] = diag[r];
            d_coefs[r][c] = diag[r + rows / 2]; // HH detail
            diag_pred_map[r][c] = diag_predictors[r];
        }
    }

    // store HL (top right), LH (bottom left), and HH (bottom right)
    for (int r = 0; r < rows / 2; ++r) {
        for (int c = 0; c < cols / 2; ++c) {
            img[r][c + cols / 2] = h_coefs[r][c];            // HL
            img[r + rows / 2][c] = v_coefs[r][c];            // LH
            img[r + rows / 2][c + cols / 2] = d_coefs[r][c]; // HH
        }
    }
}

/*******************************************************************************
 *                               AWT Reconstruction                            *
 *******************************************************************************/

// Reconstruct 1D AWT
void reconstruct_awt_1d(std::vector<float> &data, const std::vector<float> &predictors) {
    int n = data.size() / 2;
    std::vector<float> approx(n), coefs(n);
    for (int i = 0; i < n; ++i) { //[approx | detail]
        approx[i] = data[i];
        coefs[i] = data[i + n];
    }

// Undo update step
#pragma omp parallel for
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
#pragma omp parallel for
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

// Reconstruct 2D AWT
void reconstruct_awt_2d(Matrix &img,
                        const Matrix &row_pred_map,
                        const Matrix &col_pred_map,
                        const Matrix &diag_pred_map) {
    int rows = img.size();
    int cols = img[0].size();

// diagonal reconstruction
#pragma omp parallel for
    for (int c = 0; c < cols / 2; ++c) {
        std::vector<float> diag(rows);
        std::vector<float> diag_predictors(rows / 2);

        for (int r = 0; r < rows / 2; ++r) {
            diag[r] = img[r][c + cols / 2];                       // HL (approx)
            diag[r + rows / 2] = img[r + rows / 2][c + cols / 2]; // HH (detail)
            diag_predictors[r] = diag_pred_map[r][c];
        }

        reconstruct_awt_1d(diag, diag_predictors);

        for (int r = 0; r < rows; ++r) {
            img[r][c + cols / 2] = diag[r]; // reconstructed HL
        }
    }

// column wise reconstruction
#pragma omp parallel for
    for (int c = 0; c < cols / 2; ++c) {
        std::vector<float> col(rows);
        std::vector<float> col_predictors(rows / 2);

        for (int r = 0; r < rows / 2; ++r) {
            col[r] = img[r][c];                       // LL (approx)
            col[r + rows / 2] = img[r + rows / 2][c]; // LH (detail)
            col_predictors[r] = col_pred_map[r][c];
        }

        reconstruct_awt_1d(col, col_predictors);

        for (int r = 0; r < rows; ++r) {
            img[r][c] = col[r]; // reconstructed LL
        }
    }

// row wise reconstruction
#pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        std::vector<float> row(cols);
        std::vector<float> row_predictors(cols / 2);

        for (int c = 0; c < cols / 2; ++c) {
            row[c] = img[r][c];                       // LL (approx)
            row[c + cols / 2] = img[r][c + cols / 2]; // HL (detail)
            row_predictors[c] = row_pred_map[r][c];
        }

        reconstruct_awt_1d(row, row_predictors);

        for (int c = 0; c < cols; ++c) {
            img[r][c] = row[c];
        }
    }
}

Matrix load_grayscale_image(const std::string &filename) {
    int width, height, channels;
    unsigned char *img_data = stbi_load(filename.c_str(), &width, &height, &channels, 1); // force grayscale

    if (!img_data) {
        std::cerr << "Failed to load image: " << filename << "\n";
        exit(EXIT_FAILURE);
    }

    Matrix img(height, std::vector<float>(width));
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            img[y][x] = static_cast<float>(img_data[y * width + x]);

    stbi_image_free(img_data);
    return img;
}

void save_grayscale_image(const std::string &filename, const Matrix &img) {
    int height = img.size();
    int width = img[0].size();

    std::vector<unsigned char> out_data(width * height);

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            out_data[y * width + x] = static_cast<unsigned char>(
                std::clamp(img[y][x], 0.0f, 255.0f));

    if (!stbi_write_png(filename.c_str(), width, height, 1, out_data.data(), width)) {
        std::cerr << "Failed to write image: " << filename << "\n";
        exit(EXIT_FAILURE);
    }
}

void cropCompressed(const std::string &filename, const Matrix &img) {
    int height = img.size();
    int width = img[0].size();

    int outHeight = (height + 1) / 2;
    int outWidth = (width + 1) / 2;

    std::vector<unsigned char> out_data(outWidth * outHeight);

    for (int y = 0; y < outHeight; ++y)
        for (int x = 0; x < outWidth; ++x)
            out_data[y * outWidth + x] = static_cast<unsigned char>(
                std::clamp(img[y][x], 0.0f, 255.0f));

    if (!stbi_write_png(filename.c_str(), outWidth, outHeight, 1, out_data.data(), outWidth)) {
        std::cerr << "Failed to write image: " << filename << "\n";
        exit(EXIT_FAILURE);
    }
}

int main() {
    Matrix img = load_grayscale_image("cameraman.png");
    Matrix original = img;
    std::chrono::steady_clock::time_point reconst_start;

    // Prepare coefficient and predictor maps
    int rows = img.size(), cols = img[0].size();
    Matrix h(rows, std::vector<float>(cols / 2));
    Matrix v(rows / 2, std::vector<float>(cols));
    Matrix d(rows / 2, std::vector<float>(cols / 2));
    Matrix row_preds(rows, std::vector<float>(cols / 2));
    Matrix col_preds(rows / 2, std::vector<float>(cols));
    Matrix diag_preds(rows / 2, std::vector<float>(cols / 2));

    // Transform
    awt_2d(img, h, v, d, row_preds, col_preds, diag_preds);

    const auto transform_start = std::chrono::steady_clock::now();

    // Threshold detail coefficients for compression
    apply_threshold(h, 5.0f);
    apply_threshold(v, 5.0f);
    apply_threshold(d, 5.0f);

    // Save compressed image
    save_grayscale_image("compressed.png", img);
    save_image_to_file("compressed.txt", img);

    cropCompressed("compressCropped.png", img);

    const double transformation_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - transform_start).count();
    std::cout << "\033[31mTransformation time (sec): " << std::fixed << std::setprecision(10) << transformation_time << "\033[0m\n";

    reconst_start = std::chrono::steady_clock::now();

    // Reconstruct
    reconstruct_awt_2d(img, row_preds, col_preds, diag_preds);

    const double reconstruction_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - reconst_start).count();
    std::cout << "\033[34mReconstruction time (sec): " << std::fixed << std::setprecision(10) << reconstruction_time << "\033[0m\n";

    save_grayscale_image("reconstructed.png", img);
    save_image_to_file("reconstructed.txt", img);
    return 0;
}
