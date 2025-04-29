#pragma once

#include <algorithm>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<float>>;

// Clamp the boundary for predictions
inline int clamp(int x, int min, int max) {
    return std::max(min, std::min(max, x));
}

// Thresholding
void apply_threshold(Matrix &coeffs, float threshold);

// Image processing
void load_image_from_file(const std::string &filename, Matrix &img);
void save_image_to_file(const std::string &filename, const Matrix &img);
Matrix load_grayscale_image(const std::string &filename);
void save_grayscale_image(const std::string &filename, const Matrix &img);
void cropCompressed(const std::string &filename, const Matrix &img);
