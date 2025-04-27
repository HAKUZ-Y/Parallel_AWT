#include "metrics.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <mpi.h>

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

/*******************************************************************************
 *                               AWT Transformation                            *
 *******************************************************************************/

// Apply 1D AWT, stores [approx | detail] in a vector
void awt_1d(std::vector<float> &data, std::vector<float> &coefs, std::vector<float> &predictors) {
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
    std::vector<float> result(data.size()); // buffer
    for (int i = 0; i < n; ++i) {
        result[i] = data[2 * i];
        const auto &update_filter = AWT_UPDATES[predictors[i]];
        int update_filter_len = update_filter.size();
        int start = i - update_filter_len / 2;
        float update_sum = 0.0f;
        for (int j = 0; j < update_filter_len; ++j) {
            int idx = clamp(start + j, 0, n - 1); // padding for edges
            update_sum += update_filter[j] * coefs[idx];
        }
        result[i] += update_sum;  // update approx coefficient
        result[i + n] = coefs[i]; // detail in the second half
    }
    data = result;
}

// 2D AWT
void awt_transform_2d_mpi(Matrix &img,
                          Matrix &h_coefs,
                          Matrix &v_coefs,
                          Matrix &d_coefs,
                          Matrix &row_pred_map,
                          Matrix &col_pred_map,
                          Matrix &diag_pred_map,
                          int pid,
                          int nproc) {
    int rows = img.size();
    int cols = img[0].size();

    // Row distribution for horizontal pass
    int row_chunk_size = rows / nproc;
    int row_remainder = rows % nproc;

    // Each process handles its assigned rows
    std::vector<int> row_counts(nproc), row_displs(nproc);
    std::vector<int> counts_h(nproc), displs_h(nproc);
    for (int i = 0; i < nproc; ++i) {
        row_counts[i] = row_chunk_size + (i < row_remainder ? 1 : 0);
        row_displs[i] = (i == 0) ? 0 : row_displs[i - 1] + row_counts[i - 1];
        counts_h[i] = row_counts[i] * (cols / 2);
        displs_h[i] = row_displs[i] * (cols / 2);
    }

    int local_rows = row_counts[pid];
    int start_row = row_displs[pid];

    // local horizontal buffers
    std::vector<float> local_LL(local_rows * (cols / 2));
    std::vector<float> local_HL(local_rows * (cols / 2));
    std::vector<float> local_row_pred(local_rows * (cols / 2));
    std::vector<float> gathered_LL(rows * (cols / 2));
    std::vector<float> gathered_HL(rows * (cols / 2));
    std::vector<float> gathered_row_pred(rows * (cols / 2));

    // horizontal transform
    for (int r = 0; r < local_rows; ++r) {
        std::vector<float> row = img[start_row + r];
        std::vector<float> row_coef, row_pred;

        awt_1d(row, row_coef, row_pred);

        for (int c = 0; c < cols / 2; ++c) {
            local_LL[r * (cols / 2) + c] = row[c];
            local_HL[r * (cols / 2) + c] = row_coef[c];
            local_row_pred[r * (cols / 2) + c] = row_pred[c];
        }
    }

    MPI_Allgatherv(local_LL.data(), counts_h[pid], MPI_FLOAT,
                   gathered_LL.data(), counts_h.data(), displs_h.data(), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(local_HL.data(), counts_h[pid], MPI_FLOAT,
                   gathered_HL.data(), counts_h.data(), displs_h.data(), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(local_row_pred.data(), counts_h[pid], MPI_FLOAT,
                   gathered_row_pred.data(), counts_h.data(), displs_h.data(), MPI_FLOAT, MPI_COMM_WORLD);

    // Fill img and h_coefs
    h_coefs.resize(rows, std::vector<float>(cols / 2));
    row_pred_map.resize(rows, std::vector<float>(cols / 2));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols / 2; ++c) {
            img[r][c] = gathered_LL[r * (cols / 2) + c];
            h_coefs[r][c] = gathered_HL[r * (cols / 2) + c];
            row_pred_map[r][c] = gathered_row_pred[r * (cols / 2) + c];
        }

    // ########################################################################
    MPI_Barrier(MPI_COMM_WORLD);
    // ########################################################################

    // vertical transform
    int half_rows = rows / 2;
    int half_cols = cols / 2;

    int col_chunk_size = half_cols / nproc;
    int col_remainder = half_cols % nproc;

    std::vector<int> col_counts(nproc), col_displs(nproc);
    std::vector<int> counts_v(nproc), displs_v(nproc);
    for (int i = 0; i < nproc; ++i) {
        col_counts[i] = col_chunk_size + (i < col_remainder ? 1 : 0);
        col_displs[i] = (i == 0) ? 0 : col_displs[i - 1] + col_counts[i - 1];
        counts_v[i] = half_rows * col_counts[i];
        displs_v[i] = half_rows * col_displs[i];
    }

    int local_cols = col_counts[pid];
    int start_col = col_displs[pid];

    // local vertical buffers
    std::vector<float> local_LL_v(half_rows * local_cols);
    std::vector<float> local_v(half_rows * local_cols);
    std::vector<float> local_col_pred(half_rows * local_cols);
    std::vector<float> gathered_LL_v(half_rows * half_cols);
    std::vector<float> gathered_v(half_rows * half_cols);
    std::vector<float> gathered_col_pred(half_rows * half_cols);

    for (int c = 0; c < local_cols; ++c) {
        int global_c = start_col + c;
        std::vector<float> col(rows);
        std::vector<float> col_coef, col_pred;

        for (int r = 0; r < rows; ++r)
            col[r] = img[r][global_c];

        awt_1d(col, col_coef, col_pred);

        for (int r = 0; r < half_rows; ++r) {
            int idx = c * half_rows + r;
            local_LL_v[idx] = col[r];
            local_v[idx] = col_coef[r];
            local_col_pred[idx] = col_pred[r];
        }
    }

    MPI_Allgatherv(local_LL_v.data(), counts_v[pid], MPI_FLOAT,
                   gathered_LL_v.data(), counts_v.data(), displs_v.data(), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(local_v.data(), counts_v[pid], MPI_FLOAT,
                   gathered_v.data(), counts_v.data(), displs_v.data(), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(local_col_pred.data(), counts_v[pid], MPI_FLOAT,
                   gathered_col_pred.data(), counts_v.data(), displs_v.data(), MPI_FLOAT, MPI_COMM_WORLD);

    v_coefs.resize(half_rows, std::vector<float>(half_cols));
    col_pred_map.resize(half_rows, std::vector<float>(half_cols));
    for (int c = 0; c < half_cols; ++c)
        for (int r = 0; r < half_rows; ++r) {
            int idx = c * half_rows + r;
            img[r][c] = gathered_LL_v[idx];
            v_coefs[r][c] = gathered_v[idx];
            col_pred_map[r][c] = gathered_col_pred[idx];
        }

    // ########################################################################
    MPI_Barrier(MPI_COMM_WORLD);
    // ########################################################################

    // diagonal transform
    // local diagonal buffers
    std::vector<float> local_LL_d(half_rows * local_cols);
    std::vector<float> local_d(half_rows * local_cols);
    std::vector<float> local_diag_pred(half_rows * local_cols);
    std::vector<float> gathered_LL_d(half_rows * half_cols);
    std::vector<float> gathered_d(half_rows * half_cols);
    std::vector<float> gathered_diag_pred(half_rows * half_cols);

    for (int c = 0; c < local_cols; ++c) {
        int global_c = start_col + c;
        std::vector<float> diag(rows);
        std::vector<float> diag_coef, diag_pred;

        for (int r = 0; r < rows; ++r)
            diag[r] = h_coefs[r][global_c];

        awt_1d(diag, diag_coef, diag_pred);

        for (int r = 0; r < half_rows; ++r) {
            int idx = c * half_rows + r;
            local_LL_d[idx] = diag[r];
            local_d[idx] = diag_coef[r];
            local_diag_pred[idx] = diag_pred[r];
        }
    }

    MPI_Allgatherv(local_LL_d.data(), counts_v[pid], MPI_FLOAT,
                   gathered_LL_d.data(), counts_v.data(), displs_v.data(), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(local_d.data(), counts_v[pid], MPI_FLOAT,
                   gathered_d.data(), counts_v.data(), displs_v.data(), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(local_diag_pred.data(), counts_v[pid], MPI_FLOAT,
                   gathered_diag_pred.data(), counts_v.data(), displs_v.data(), MPI_FLOAT, MPI_COMM_WORLD);

    d_coefs.resize(half_rows, std::vector<float>(half_cols));
    diag_pred_map.resize(half_rows, std::vector<float>(half_cols));
    for (int c = 0; c < half_cols; ++c)
        for (int r = 0; r < half_rows; ++r) {
            int idx = c * half_rows + r;
            h_coefs[r][c] = gathered_LL_d[idx];
            d_coefs[r][c] = gathered_d[idx];
            diag_pred_map[r][c] = gathered_diag_pred[idx];
        }

    MPI_Barrier(MPI_COMM_WORLD);

    // update image
    for (int r = 0; r < half_rows; ++r) {
        for (int c = 0; c < half_cols; ++c) {
            img[r][c + half_cols] = h_coefs[r][c];
            img[r + half_rows][c] = v_coefs[r][c];
            img[r + half_rows][c + half_cols] = d_coefs[r][c];
        }
    }
}

int main(int argc, char *argv[]) {
    int pid;
    int nproc;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    Matrix img;
    int row, col;

    std::string image_name = argv[1];
    std::string image_filename = image_name + ".png";

    if (pid == 0) {
        img = load_grayscale_image(image_filename);
        row = img.size();
        col = img[0].size();
    }

    MPI_Bcast(&row, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&col, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (pid != 0) {
        img.resize(row, std::vector<float>(col));
    }

    std::vector<float> flat_img;
    if (pid == 0) {
        flat_img.reserve(row * col);
        for (const auto &r : img)
            flat_img.insert(flat_img.end(), r.begin(), r.end());
    }

    flat_img.resize(row * col);
    MPI_Bcast(flat_img.data(), row * col, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (pid != 0) {
        for (int i = 0; i < row; ++i) {
            img[i] = std::vector<float>(flat_img.begin() + i * col, flat_img.begin() + (i + 1) * col);
        }
    }

    // Allocate coefficient and predictor matrices
    Matrix h(row, std::vector<float>(col / 2));
    Matrix v(row / 2, std::vector<float>(col / 2));
    Matrix d(row / 2, std::vector<float>(col / 2));
    Matrix row_preds(row, std::vector<float>(col / 2));
    Matrix col_preds(row / 2, std::vector<float>(col / 2));
    Matrix diag_preds(row / 2, std::vector<float>(col / 2));

    const auto compute_start = std::chrono::steady_clock::now();

    // Transform
    awt_transform_2d_mpi(img, h, v, d, row_preds, col_preds, diag_preds, pid, nproc);

    // Apply threshold + save image
    if (pid == 0) {
        apply_threshold(h, 5.0f);
        apply_threshold(v, 5.0f);
        apply_threshold(d, 5.0f);

        save_grayscale_image("compressed.png", img);
        save_image_to_file("compressed.txt", img);
        cropCompressed("compressCropped.png", img);

        const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
        std::cout << "\033[31mComputation time (sec): " << std::fixed << std::setprecision(10) << compute_time << "\033[0m\n";
    }

    MPI_Finalize();
    return 0;
}

// next step: pipeline where data flows through processors that each perform a different stage of the transform