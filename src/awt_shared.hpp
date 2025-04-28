#pragma once
#include "awt_common.hpp"
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

/*******************************************************************************
 *                               AWT Transformation                            *
 *******************************************************************************/

// Apply 1D AWT
void awt_1d_shared(std::vector<double> &data, std::vector<double> &coefs, std::vector<int> &predictors) {
    int n = data.size() / 2;
    coefs.resize(n);
    predictors.resize(n);

// Predict step: predict odd indices from even indices
#pragma omp parallel for default(shared) schedule(static)
    for (int i = 0; i < n; ++i) {
        int odd_idx = 2 * i + 1;
        double min_err = INF;
        double min_coef = 0.0;
        int pred_index = 0;

        // find the best predictor adaptively
        // TODO: optimize
        for (size_t p = 0; p < AWT_PREDICTORS.size(); ++p) {
            const auto &filter = AWT_PREDICTORS[p];
            int len = filter.size();
            int start_index = odd_idx - len + 1;
            double pred_val = 0.0;

            for (int j = 0; j < len; ++j) {
                int idx = mirror(start_index + j * 2, data.size());
                pred_val += filter[j] * data[idx];
            }

            double err = std::abs(data[odd_idx] - pred_val);
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

    // Update step
    // Need to match with the reverse reconstruction
    std::vector<double> result(data.size());
#pragma omp parallel for default(shared) schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        result[2 * i] = data[2 * i]; // start from even

        const auto &update_filter = AWT_UPDATES[predictors[i]];
        int len = update_filter.size();
        int start = i - (len / 2 - 1);

        double update_sum = 0.0;
        for (int j = 0; j < len; ++j) {
            int idx = mirror(start + j, n);
            update_sum += update_filter[j] * coefs[idx];
        }

        result[2 * i] += update_sum;         // updated even
        result[2 * i + 1] = data[2 * i + 1]; // untouched odd
    }
    data = result;
}

// Apply 2D AWT
void awt_2d_shared(Matrix &img,
                   Matrix &row_pred_map,
                   Matrix &col_pred_map,
                   Matrix &diag_pred_map,
                   double threshold) {
    int rows = img.size();
    int cols = img[0].size();
    int half_rows = rows / 2;
    int half_cols = cols / 2;

    // row wise transform
#pragma omp parallel for default(shared) schedule(static)
    for (int r = 0; r < rows; ++r) {
        std::vector<double> row = img[r];
        std::vector<double> row_coef;
        std::vector<int> row_predictors;
        awt_1d_shared(row, row_coef, row_predictors);

        for (int c = 0; c < half_cols; ++c) {
            // LL region: approximation image
            img[r][c] = row[2 * c];
            // HL region: horizontal coefficients for reconstruction
            // compress the coefficients by the threshold
            img[r][half_cols + c] = (std::abs(row_coef[c]) < threshold) ? 0.0 : row_coef[c];
            // row predictor map
            row_pred_map[r][c] = row_predictors[c];
        }
    }

// column wise transform
#pragma omp parallel for default(shared) schedule(static)
    for (int c = 0; c < half_cols; ++c) {
        std::vector<double> col(rows);
        std::vector<double> col_coef;
        std::vector<int> col_predictors;

        // TODO: optimize?
        for (int r = 0; r < rows; ++r) {
            col[r] = img[r][c];
        }
        awt_1d_shared(col, col_coef, col_predictors);

        for (int r = 0; r < half_rows; ++r) {
            // LL region: approximation image
            img[r][c] = col[2 * r];
            // LH region: vertical coefficients for reconstruction
            img[r + half_rows][c] = (std::abs(col_coef[r]) < threshold) ? 0.0 : col_coef[r];
            // column predictor map
            col_pred_map[r][c] = col_predictors[r];
        }

        std::vector<double> diag(rows);
        std::vector<double> diag_coef;
        std::vector<int> diag_predictors;

        // get diagonal coefficients
        for (int r = 0; r < rows; ++r) {
            diag[r] = img[r][c + half_cols];
        }

        awt_1d_shared(diag, diag_coef, diag_predictors);

        for (int r = 0; r < half_rows; ++r) {
            // HH region: diagonal coefficients for reconstruction
            img[r + half_rows][c + half_cols] = (std::abs(diag_coef[r]) < threshold) ? 0.0 : diag_coef[r];
            // diagonal predictor map
            diag_pred_map[r][c] = diag_predictors[r];
        }
    }
}

// Apply adaptive lifting recursively (multi-level)
void awt_multi_level_shared(Matrix &img, int levels, double threshold,
                            std::vector<Matrix> &row_pred_maps,
                            std::vector<Matrix> &col_pred_maps,
                            std::vector<Matrix> &diag_pred_maps) {

    for (int level = 0; level < levels; ++level) {

        int rows = img.size() >> level;
        int cols = img[0].size() >> level;

        // extract LL matrix from last level
        Matrix approx_img(rows, std::vector<double>(cols));

// copy the image
#pragma omp parallel for default(shared) schedule(static)
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                approx_img[r][c] = img[r][c];
            }
        }

        // apply 2d AWT
        Matrix row_pred_map(rows, std::vector<double>(cols / 2));
        Matrix col_pred_map(rows / 2, std::vector<double>(cols));
        Matrix diag_pred_map(rows / 2, std::vector<double>(cols / 2));
        awt_2d_shared(approx_img, row_pred_map, col_pred_map, diag_pred_map, threshold);

        // coefficents for reconstruction
        row_pred_maps.push_back(row_pred_map);
        col_pred_maps.push_back(col_pred_map);
        diag_pred_maps.push_back(diag_pred_map);

// update the image with the approximation
#pragma omp parallel for default(shared) schedule(static)
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                img[r][c] = approx_img[r][c];
            }
        }
    }
}

/*******************************************************************************
 *                               AWT Reconstruction                            *
 *******************************************************************************/

// Reconstruct 1D AWT
void reconst_awt_1d_shared(std::vector<double> &data,
                           const std::vector<double> &coefs,
                           const std::vector<int> &predictors) {
    int n = coefs.size();

    // Undo update step
    // TODO: match the forward side if changed
    for (int i = 0; i < n; ++i) {
        const auto &update_filter = AWT_UPDATES[predictors[i]];
        int len = update_filter.size();
        int start = i - (len / 2 - 1);

        double update_sum = 0.0;
        for (int j = 0; j < len; ++j) {
            int idx = mirror(start + j, n);
            update_sum += update_filter[j] * coefs[idx];
        }

        data[2 * i] -= update_sum;
    }

    // Reverse predict step
    for (int i = 0; i < n; ++i) {
        int odd_idx = 2 * i + 1;

        // get the predictor used
        int pred_index = predictors[i];
        const auto &filter = AWT_PREDICTORS[pred_index];

        int len = filter.size();
        int start_index = odd_idx - len + 1;
        double pred_val = 0.0;

        for (int j = 0; j < len; ++j) {
            int idx = mirror(start_index + j * 2, data.size());
            pred_val += filter[j] * data[idx];
        }

        // Reconstruct odd sample
        data[odd_idx] = pred_val + coefs[i];
    }
}

void reconst_awt_2d_shared(Matrix &img,
                           int rows, int cols,
                           const Matrix &row_pred_map,
                           const Matrix &col_pred_map,
                           const Matrix &diag_pred_map) {
    int half_rows = rows / 2;
    int half_cols = cols / 2;

    Matrix temp_img(rows, std::vector<double>(cols));

    // HH: Reconstruct diagonal coefficients
    for (int c = 0; c < half_cols; ++c) {
        std::vector<double> diag(rows);
        std::vector<double> diag_coef(half_rows);
        std::vector<int> diag_predictors(half_rows);

        for (int r = 0; r < half_rows; ++r) {
            diag[2 * r] = img[r][c + half_cols];
            diag_coef[r] = img[r + half_rows][c + half_cols];
            diag_predictors[r] = diag_pred_map[r][c];
        }

        reconst_awt_1d_shared(diag, diag_coef, diag_predictors);

        for (int r = 0; r < rows; ++r) {
            temp_img[r][c + half_cols] = diag[r];
        }
    }

    // LH: Reconstruct vertical coefficients
    for (int c = 0; c < half_cols; ++c) {
        std::vector<double> col(rows);
        std::vector<double> col_coef(half_rows);
        std::vector<int> col_predictors(half_rows);

        for (int r = 0; r < half_rows; ++r) {
            col[2 * r] = img[r][c];
            col_coef[r] = img[r + half_rows][c];
            col_predictors[r] = col_pred_map[r][c];
        }

        reconst_awt_1d_shared(col, col_coef, col_predictors);

        for (int r = 0; r < rows; ++r) {
            temp_img[r][c] = col[r];
        }
    }

    // HL: Reconstruct horizontal coefficients
    for (int r = 0; r < rows; ++r) {
        std::vector<double> row(cols);
        std::vector<double> row_coef(half_cols);
        std::vector<int> row_predictors(half_cols);

        for (int c = 0; c < half_cols; ++c) {
            row[2 * c] = temp_img[r][c];
            row_coef[c] = temp_img[r][c + half_cols];
            row_predictors[c] = row_pred_map[r][c];
        }

        reconst_awt_1d_shared(row, row_coef, row_predictors);

        for (int c = 0; c < cols; ++c) {
            img[r][c] = row[c];
        }
    }
}

// Reconstruct the image from coefficients
void reconst_awt_shared(Matrix &img, int levels,
                        const std::vector<Matrix> &row_pred_maps,
                        const std::vector<Matrix> &col_pred_maps,
                        const std::vector<Matrix> &diag_pred_maps) {

    for (int level = levels - 1; level >= 0; --level) {

        int rows = img.size() >> level;
        int cols = img[0].size() >> level;
        int half_rows = rows / 2;
        int half_cols = cols / 2;

        // Reconstruct 2D AWT
        reconst_awt_2d_shared(img,
                              rows, cols,
                              row_pred_maps[level],
                              col_pred_maps[level],
                              diag_pred_maps[level]);
    }
}