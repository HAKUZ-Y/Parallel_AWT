#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

using Matrix = std::vector<std::vector<float>>;

// Mean Squared Error (MSE)
inline float compute_mse(const Matrix& img1, const Matrix& img2) {
    int rows = img1.size();
    int cols = img1[0].size();
    float mse = 0.0f;

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            float diff = img1[i][j] - img2[i][j];
            mse += diff * diff;
        }

    return mse / (rows * cols);
}

// Structural Similarity Index Measure (SSIM) simplified
inline float compute_ssim(const Matrix& img1, const Matrix& img2) {
    constexpr float C1 = 6.5025f, C2 = 58.5225f;

    int rows = img1.size();
    int cols = img1[0].size();

    float mean_x = 0, mean_y = 0, sigma_x = 0, sigma_y = 0, sigma_xy = 0;

    // Compute means
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            mean_x += img1[i][j];
            mean_y += img2[i][j];
        }
    mean_x /= (rows * cols);
    mean_y /= (rows * cols);

    // Compute variances and covariance
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            float dx = img1[i][j] - mean_x;
            float dy = img2[i][j] - mean_y;
            sigma_x += dx * dx;
            sigma_y += dy * dy;
            sigma_xy += dx * dy;
        }
    sigma_x /= (rows * cols - 1);
    sigma_y /= (rows * cols - 1);
    sigma_xy /= (rows * cols - 1);

    // Compute SSIM
    float numerator = (2 * mean_x * mean_y + C1) * (2 * sigma_xy + C2);
    float denominator = (mean_x * mean_x + mean_y * mean_y + C1) * (sigma_x + sigma_y + C2);

    return numerator / denominator;
}
