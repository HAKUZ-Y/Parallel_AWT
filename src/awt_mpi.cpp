#include "helper.hpp"
#include "metrics.hpp"
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

#include <omp.h>

using Matrix = std::vector<std::vector<float>>;
constexpr float INF = 1e9f;
// Predictors
// TODO: find more with some sources
// Assume odd length for now
const Matrix AWT_PREDICTORS = {
    {0.5f, 0.5f},                    // avg of neighbors
    {-0.25f, 0.75f, 0.75f, -0.25f}}; // a more complex filter

const Matrix AWT_UPDATES = {           // to preserve structure around 2 * i
    {0.25f, 0.25f},                    // for the first predictor
    {-0.125f, 0.25f, 0.25f, -0.125f}}; // for the more complex predictor

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
    std::vector<int> row_counts(nproc), row_displs(nproc), counts_h(nproc), displs_h(nproc);
    for (int i = 0; i < nproc; ++i) {
        row_counts[i] = row_chunk_size + (i < row_remainder ? 1 : 0);
        row_displs[i] = (i == 0) ? 0 : row_displs[i - 1] + row_counts[i - 1];
        counts_h[i] = row_counts[i] * (cols / 2) * 3; // times 3 because packed
        displs_h[i] = (i == 0) ? 0 : displs_h[i - 1] + counts_h[i - 1];
    }

    int local_rows = row_counts[pid];
    int start_row = row_displs[pid];

    // --- Horizontal transform ---
    std::vector<float> local_horizontal_pack(local_rows * (cols / 2) * 3);

    // #pragma omp parallel for default(shared) schedule(static)
    for (int r = 0; r < local_rows; ++r) {
        std::vector<float> row = img[start_row + r];
        std::vector<float> row_coef, row_pred;
        awt_1d(row, row_coef, row_pred);

        for (int c = 0; c < cols / 2; ++c) {
            int idx = (r * (cols / 2) + c) * 3;
            local_horizontal_pack[idx] = row[c];
            local_horizontal_pack[idx + 1] = row_coef[c];
            local_horizontal_pack[idx + 2] = row_pred[c];
        }
    }

    std::vector<float> gathered_horizontal_pack(rows * (cols / 2) * 3);

    MPI_Allgatherv(local_horizontal_pack.data(), counts_h[pid], MPI_FLOAT,
                   gathered_horizontal_pack.data(), counts_h.data(), displs_h.data(), MPI_FLOAT, MPI_COMM_WORLD);

    // Unpack gathered horizontal data
    h_coefs.resize(rows, std::vector<float>(cols / 2));
    row_pred_map.resize(rows, std::vector<float>(cols / 2));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols / 2; ++c) {
            int idx = (r * (cols / 2) + c) * 3;
            img[r][c] = gathered_horizontal_pack[idx];
            h_coefs[r][c] = gathered_horizontal_pack[idx + 1];
            row_pred_map[r][c] = gathered_horizontal_pack[idx + 2];
        }
    }

    // --- Vertical transform ---
    int half_rows = rows / 2;
    int half_cols = cols / 2;

    int col_chunk_size = half_cols / nproc;
    int col_remainder = half_cols % nproc;

    std::vector<int> col_counts(nproc), col_displs(nproc), counts_v(nproc), displs_v(nproc);
    for (int i = 0; i < nproc; ++i) {
        col_counts[i] = col_chunk_size + (i < col_remainder ? 1 : 0);
        col_displs[i] = (i == 0) ? 0 : col_displs[i - 1] + col_counts[i - 1];
        counts_v[i] = half_rows * col_counts[i] * 3; // times 3 because packed
        displs_v[i] = (i == 0) ? 0 : displs_v[i - 1] + counts_v[i - 1];
    }

    int local_cols = col_counts[pid];
    int start_col = col_displs[pid];

    std::vector<float> local_vertical_pack(half_rows * local_cols * 3);

    // #pragma omp parallel for default(shared) schedule(static)
    for (int c = 0; c < local_cols; ++c) {
        int global_c = start_col + c;
        std::vector<float> col(rows);
        std::vector<float> col_coef, col_pred;

        for (int r = 0; r < rows; ++r)
            col[r] = img[r][global_c];

        awt_1d(col, col_coef, col_pred);

        for (int r = 0; r < half_rows; ++r) {
            int idx = (c * half_rows + r) * 3;
            local_vertical_pack[idx] = col[r];
            local_vertical_pack[idx + 1] = col_coef[r];
            local_vertical_pack[idx + 2] = col_pred[r];
        }
    }

    std::vector<float> gathered_vertical_pack(half_rows * half_cols * 3);

    MPI_Allgatherv(local_vertical_pack.data(), counts_v[pid], MPI_FLOAT,
                   gathered_vertical_pack.data(), counts_v.data(), displs_v.data(), MPI_FLOAT, MPI_COMM_WORLD);

    // Unpack gathered vertical data
    v_coefs.resize(half_rows, std::vector<float>(half_cols));
    col_pred_map.resize(half_rows, std::vector<float>(half_cols));
    for (int c = 0; c < half_cols; ++c) {
        for (int r = 0; r < half_rows; ++r) {
            int idx = (c * half_rows + r) * 3;
            img[r][c] = gathered_vertical_pack[idx];
            v_coefs[r][c] = gathered_vertical_pack[idx + 1];
            col_pred_map[r][c] = gathered_vertical_pack[idx + 2];
        }
    }

    // --- Diagonal transform ---
    std::vector<float> local_diagonal_pack(half_rows * local_cols * 3);

    // #pragma omp parallel for default(shared) schedule(static)
    for (int c = 0; c < local_cols; ++c) {
        int global_c = start_col + c;
        std::vector<float> diag(rows);
        std::vector<float> diag_coef, diag_pred;

        for (int r = 0; r < rows; ++r)
            diag[r] = h_coefs[r][global_c];

        awt_1d(diag, diag_coef, diag_pred);

        for (int r = 0; r < half_rows; ++r) {
            int idx = (c * half_rows + r) * 3;
            local_diagonal_pack[idx] = diag[r];
            local_diagonal_pack[idx + 1] = diag_coef[r];
            local_diagonal_pack[idx + 2] = diag_pred[r];
        }
    }

    std::vector<float> gathered_diagonal_pack(half_rows * half_cols * 3);

    MPI_Allgatherv(local_diagonal_pack.data(), counts_v[pid], MPI_FLOAT,
                   gathered_diagonal_pack.data(), counts_v.data(), displs_v.data(), MPI_FLOAT, MPI_COMM_WORLD);

    // Unpack gathered diagonal data
    d_coefs.resize(half_rows, std::vector<float>(half_cols));
    diag_pred_map.resize(half_rows, std::vector<float>(half_cols));
    for (int c = 0; c < half_cols; ++c) {
        for (int r = 0; r < half_rows; ++r) {
            int idx = (c * half_rows + r) * 3;
            h_coefs[r][c] = gathered_diagonal_pack[idx];
            d_coefs[r][c] = gathered_diagonal_pack[idx + 1];
            diag_pred_map[r][c] = gathered_diagonal_pack[idx + 2];
        }
    }

    // --- Final image assembly ---
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
        const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
        std::cout << "\033[31mComputation time (sec): " << std::fixed << std::setprecision(10) << compute_time << "\033[0m\n";

        apply_threshold(h, 5.0f);
        apply_threshold(v, 5.0f);
        apply_threshold(d, 5.0f);
        // save_grayscale_image("compressed.png", img);
        // save_image_to_file("compressed.txt", img);
        // cropCompressed("compressCropped.png", img);
    }

    MPI_Finalize();
    return 0;
}

// next step: pipeline where data flows through processors that each perform a different stage of the transform