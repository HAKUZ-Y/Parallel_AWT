#include "awt_common.hpp"
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


// BASELINE Version

/*******************************************************************************
 *                               AWT Transformation                            *
 *******************************************************************************/

// Apply 1D AWT
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
        // printf("------ODD idx: %d\n", odd_idx);
        // find the best predictor adaptively
        // TODO: optimize
        for (size_t p = 0; p < AWT_PREDICTORS.size(); ++p) {
            const auto &filter = AWT_PREDICTORS[p];
            int len = filter.size();
            int start_index = odd_idx - len + 1;
            float pred_val = 0.0f;

            for (int j = 0; j < len; ++j) {
                int idx = mirror(start_index + j * 2, data.size());
                // printf("idx: %d\n", idx);
                pred_val += filter[j] * data[idx];
            }
            // printf("---\n");

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

    // Update step
    // TODO: is this the correct way/only way to update?
    // Need to match with the reverse reconstruction

    std::vector<float> result(data.size());
    for (int i = 0; i < n; ++i) {
        result[2 * i] = data[2 * i]; // start from even

        const auto &update_filter = AWT_UPDATES[predictors[i]];
        int len = update_filter.size();
        int start = i - (len / 2 - 1);

        float update_sum = 0.0f;
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
void awt_2d(Matrix &img, Matrix &h_coefs, Matrix &v_coefs, Matrix &d_coefs,
            Matrix &row_pred_map, Matrix &col_pred_map, Matrix &diag_pred_map) {
    int rows = img.size();
    int cols = img[0].size();

    // row wise transform
    for (int r = 0; r < rows; ++r) {
        std::vector<float> row = img[r];
        std::vector<float> row_coef;
        std::vector<int> row_predictors;
        awt_1d(row, row_coef, row_predictors);

        for (int c = 0; c < cols / 2; ++c) {
            // approximation image
            img[r][c] = row[2 * c];
            // horizontal coefficients for reconstruction
            h_coefs[r][c] = row_coef[c];
            // row predictor map
            row_pred_map[r][c] = row_predictors[c];
        }
    }

// column wise transform
    for (int c = 0; c < cols / 2; ++c) {
        std::vector<float> col(rows);
        std::vector<float> col_coef;
        std::vector<int> col_predictors;

// TODO: optimize?
        for (int r = 0; r < rows; ++r) {
            col[r] = img[r][c];
        }
        awt_1d(col, col_coef, col_predictors);

        for (int r = 0; r < rows / 2; ++r) {
            // approximation image
            img[r][c] = col[2 * r];
            // vertical coefficients for reconstruction
            v_coefs[r][c] = col_coef[r];
            // column predictor map
            col_pred_map[r][c] = col_predictors[r];
        }

        std::vector<float> diag(rows);
        std::vector<float> diag_coef;
        std::vector<int> diag_predictors;

// get diagonal coefficients
        for (int r = 0; r < rows; ++r) {
            diag[r] = h_coefs[r][c];
        }
        awt_1d(diag, diag_coef, diag_predictors);
        for (int r = 0; r < rows / 2; ++r) {
            d_coefs[r][c] = diag_coef[r];
            diag_pred_map[r][c] = diag_predictors[r];
        }
    }
}

// Apply adaptive lifting recursively (multi-level)
void multi_level_awt(Matrix &img, int levels,
                     std::vector<Matrix> &horizontal_coefs,
                     std::vector<Matrix> &vertical_coefs,
                     std::vector<Matrix> &diagonal_coefs,
                     std::vector<Matrix> &row_pred_maps,
                     std::vector<Matrix> &col_pred_maps,
                     std::vector<Matrix> &diag_pred_maps) {

    // Clear up
    horizontal_coefs.clear();
    vertical_coefs.clear();
    diagonal_coefs.clear();
    row_pred_maps.clear();

    int rows = img.size();
    int cols = img[0].size();

    for (int level = 0; level < levels; ++level) {

        // extract LL matrix from last level
        Matrix approx_img(rows, std::vector<float>(cols));

// copy the image
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                approx_img[r][c] = img[r][c];
            }
        }

        // apply 2d AWT
        Matrix h_coefs(rows, std::vector<float>(cols / 2));
        Matrix v_coefs(rows / 2, std::vector<float>(cols));
        Matrix d_coefs(rows / 2, std::vector<float>(cols / 2));
        Matrix row_pred_map(rows, std::vector<float>(cols / 2));
        Matrix col_pred_map(rows / 2, std::vector<float>(cols));
        Matrix diag_pred_map(rows / 2, std::vector<float>(cols / 2));
        awt_2d(approx_img, h_coefs, v_coefs, d_coefs,
               row_pred_map, col_pred_map, diag_pred_map);

        // coefficents for reconstruction
        horizontal_coefs.push_back(h_coefs);
        vertical_coefs.push_back(v_coefs);
        diagonal_coefs.push_back(d_coefs);
        row_pred_maps.push_back(row_pred_map);
        col_pred_maps.push_back(col_pred_map);
        diag_pred_maps.push_back(diag_pred_map);

// update the image with the approximation
        for (int r = 0; r < rows / 2; ++r) {
            for (int c = 0; c < cols / 2; ++c) {
                img[r][c] = approx_img[r][c];
            }
        }

        // update rows and cols for next level
        rows /= 2;
        cols /= 2;
    }
}

/*******************************************************************************
 *                               AWT Reconstruction                            *
 *******************************************************************************/

// Reconstruct 1D AWT
void reconstruct_awt_1d(std::vector<float> &data, const std::vector<float> &coefs, const std::vector<int> &predictors) {
    int n = coefs.size();

    // Undo update step
    // TODO: match the forward side if changed
    for (int i = 0; i < n; ++i) {
        const auto &update_filter = AWT_UPDATES[predictors[i]];
        int len = update_filter.size();
        int start = i - (len / 2 - 1);

        float update_sum = 0.0f;
        for (int j = 0; j < len; ++j) {
            int idx = mirror(start + j, n);
            update_sum += update_filter[j] * coefs[idx];
        }

        data[2 * i] -= update_sum;
    }

    // Reverse predict step
    for (int i = 0; i < n; ++i) {
        int odd_idx = 2 * i + 1;
        // printf("|||||||||||ODD idx: %d\n", odd_idx);

        // get the predictor used
        int pred_index = predictors[i];
        const auto &filter = AWT_PREDICTORS[pred_index];

        int len = filter.size();
        int start_index = odd_idx - len + 1;
        float pred_val = 0.0f;

        for (int j = 0; j < len; ++j) {
            int idx = mirror(start_index + j * 2, data.size());
            // printf("idx: %d\n", idx);
            pred_val += filter[j] * data[idx];
        }
        // printf("|||\n");

        // Reconstruct odd sample
        data[odd_idx] = pred_val + coefs[i];
    }
}

// Reconstruct 2D AWT
void reconstruct_awt_2d(Matrix &img, const Matrix &h_coefs, const Matrix &v_coefs, const Matrix &d_coefs,
                        const Matrix &row_pred_map, const Matrix &col_pred_map, const Matrix &diag_pred_map) {
    int rows = img.size();
    int cols = img[0].size();

    // Create a temporary image with the same size as the original
    Matrix temp_img = img;

    // TODO: do I need this
    // diagonal reconstruction
    // for (int c = 0; c < cols / 2; ++c) {
    //     std::vector<float> diag(rows);
    //     std::vector<float> diag_coef(rows / 2);
    //     std::vector<int> diag_predictors(rows / 2);

    //     // Setup even samples (horizontal details)
    //     for (int r = 0; r < rows / 2; ++r) {
    //         diag[2 * r] = h_coefs[r][c];
    //         diag_coef[r] = d_coefs[r][c];
    //         diag_predictors[r] = static_cast<int>(diag_pred_map[r][c]);
    //     }

    //     // Reconstruct 1D AWT
    //     reconstruct_awt_1d(diag, diag_coef, diag_predictors);

    //     // Save the reconstructed horizontal details
    //     for (int r = 0; r < rows; ++r) {
    //         temp_img[r][c] = diag[r];
    //     }
    // }

    // column wise reconstruction
    for (int c = 0; c < cols / 2; ++c) {
        std::vector<float> col(rows);
        std::vector<float> col_coef(rows / 2);
        std::vector<int> col_predictors(rows / 2);

        // TODO: optimize?
        for (int r = 0; r < rows / 2; ++r) {
            col[2 * r] = img[r][c];
            col_coef[r] = v_coefs[r][c];
            col_predictors[r] = col_pred_map[r][c];
        }

        // Reconstruct 1D AWT
        reconstruct_awt_1d(col, col_coef, col_predictors);

        for (int r = 0; r < rows; ++r) {
            temp_img[r][c] = col[r];
        }
    }

    // row wise reconstruction
    for (int r = 0; r < rows; ++r) {
        std::vector<float> row(cols);
        std::vector<float> row_coef(cols / 2);
        std::vector<int> row_predictors(cols / 2);

        for (int c = 0; c < cols / 2; ++c) {
            row[2 * c] = temp_img[r][c];
            row_coef[c] = h_coefs[r][c];
            row_predictors[c] = row_pred_map[r][c];
        }

        // Reconstruct 1D AWT
        reconstruct_awt_1d(row, row_coef, row_predictors);

        for (int c = 0; c < cols; ++c) {
            img[r][c] = row[c];
        }
    }
}

// Reconstruct the image from coefficients
void reconstruct_awt(Matrix &img, int levels,
                     const std::vector<Matrix> &horizontal_coefs,
                     const std::vector<Matrix> &vertical_coefs,
                     const std::vector<Matrix> &diagonal_coefs,
                     const std::vector<Matrix> &row_pred_maps,
                     const std::vector<Matrix> &col_pred_maps,
                     const std::vector<Matrix> &diag_pred_maps) {

    for (int level = levels - 1; level >= 0; --level) {

        int rows = img.size() / (1 << level);
        int cols = img.size() / (1 << level);

        // original image for each level
        // TODO: optimize by inplace and indices?
        Matrix original_img(rows, std::vector<float>(cols));
        for (int r = 0; r < rows / 2; ++r) {
            for (int c = 0; c < cols / 2; ++c) {
                original_img[r][c] = img[r][c];
            }
        }

        // Reconstruct 2D AWT
        reconstruct_awt_2d(original_img,
                           horizontal_coefs[level],
                           vertical_coefs[level],
                           diagonal_coefs[level],
                           row_pred_maps[level],
                           col_pred_maps[level],
                           diag_pred_maps[level]);

        // Copy the data back
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                img[r][c] = original_img[r][c];
            }
        }
    }
}

/*******************************************************************************
 *                             Apply Threshold                                 *
 *******************************************************************************/

// Apply thresholding for compression
void apply_thresholding(std::vector<Matrix> &all_details_h,
                        std::vector<Matrix> &all_details_v,
                        std::vector<Matrix> &all_details_d,
                        float threshold) {
    for (size_t level = 0; level < all_details_h.size(); ++level) {
        // Apply thresholding to each type of detail coefficients
        for (auto &row : all_details_h[level]) {
            for (auto &val : row) {
                if (std::abs(val) < threshold)
                    val = 0.0f;
            }
        }

        for (auto &row : all_details_v[level]) {
            for (auto &val : row) {
                if (std::abs(val) < threshold)
                    val = 0.0f;
            }
        }

        for (auto &row : all_details_d[level]) {
            for (auto &val : row) {
                if (std::abs(val) < threshold)
                    val = 0.0f;
            }
        }
    }
}
