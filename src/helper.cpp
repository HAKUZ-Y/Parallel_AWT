#include "helper.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/*******************************************************************************
 *                              Helper Functions                               *
 *******************************************************************************/
namespace fs = std::filesystem;

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