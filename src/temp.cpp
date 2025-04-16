#include "metrics.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <iomanip>  // For formatting output

using Matrix = std::vector<std::vector<float>>;
constexpr float INF = 1e9f;

// Predictors
// TODO: find more with some sources
// Assume odd lenght for now
const Matrix AWT_PREDICTORS = {
    {0},
    {0.5f, 0, 0.5f},
    {-0.125f, 0, 0.25f, 0.75f, 0.25f, 0, -0.125f}};

// Clamp the boundary for predictions
inline int clamp(int x, int min, int max) {
    return std::max(min, std::min(max, x));
}

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
    std::cout << "Save image " << filename << " with size "
              << img.size() << "x" << img[0].size() << std::endl;
}

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

        // find the best predictor adaptively
        // TODO: optimize
        for (size_t p = 0; p < AWT_PREDICTORS.size(); ++p) {
            const auto &filter = AWT_PREDICTORS[p];
            int len = filter.size();
            int start_index = odd_idx - len / 2;
            float pred_val = 0.0f;

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

        coefs[i] = min_coef;
        predictors[i] = pred_index;
    }

    // Update step
    // TODO: is this the correct way/only way to update?
    // Need to match with the inverse reconstruction
    for (int i = 0; i < n; ++i) {
        data[2 * i] += coefs[i] * 0.5f;
    }
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

// Apply thresholding to wavelet coefficients
void apply_thresholding(std::vector<Matrix> &horizontal_coefs,
                       std::vector<Matrix> &vertical_coefs,
                       std::vector<Matrix> &diagonal_coefs,
                       float threshold) {
    
    for (size_t level = 0; level < horizontal_coefs.size(); ++level) {
        // Process horizontal coefficients
        for (auto &row : horizontal_coefs[level]) {
            for (auto &val : row) {
                if (std::abs(val) < threshold) {
                    val = 0.0f;
                }
            }
        }
        
        // Process vertical coefficients
        for (auto &row : vertical_coefs[level]) {
            for (auto &val : row) {
                if (std::abs(val) < threshold) {
                    val = 0.0f;
                }
            }
        }
        
        // Process diagonal coefficients
        for (auto &row : diagonal_coefs[level]) {
            for (auto &val : row) {
                if (std::abs(val) < threshold) {
                    val = 0.0f;
                }
            }
        }
    }
}

// Count non-zero coefficients
int count_nonzeros(const Matrix &approx,
                  const std::vector<Matrix> &horizontal_coefs,
                  const std::vector<Matrix> &vertical_coefs,
                  const std::vector<Matrix> &diagonal_coefs) {
    
    int count = 0;
    
    // Count non-zeros in the approximation band
    for (const auto &row : approx) {
        for (const auto &val : row) {
            if (val != 0.0f) {
                count++;
            }
        }
    }
    
    // Count non-zeros in detail coefficients
    for (const auto &level_coefs : horizontal_coefs) {
        for (const auto &row : level_coefs) {
            for (const auto &val : row) {
                if (val != 0.0f) {
                    count++;
                }
            }
        }
    }
    
    for (const auto &level_coefs : vertical_coefs) {
        for (const auto &row : level_coefs) {
            for (const auto &val : row) {
                if (val != 0.0f) {
                    count++;
                }
            }
        }
    }
    
    for (const auto &level_coefs : diagonal_coefs) {
        for (const auto &row : level_coefs) {
            for (const auto &val : row) {
                if (val != 0.0f) {
                    count++;
                }
            }
        }
    }
    
    return count;
}

// Get approximation matrix from transformed image
Matrix extract_approximation(const Matrix &img, int levels) {
    int approx_size = img.size() / (1 << levels);
    Matrix approx(approx_size, std::vector<float>(approx_size));
    
    for (int r = 0; r < approx_size; ++r) {
        for (int c = 0; c < approx_size; ++c) {
            approx[r][c] = img[r][c];
        }
    }
    
    return approx;
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
    col_pred_maps.clear();
    diag_pred_maps.clear();

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
        Matrix v_coefs(rows / 2, std::vector<float>(cols / 2));
        Matrix d_coefs(rows / 2, std::vector<float>(cols / 2));
        Matrix row_pred_map(rows, std::vector<float>(cols / 2));
        Matrix col_pred_map(rows / 2, std::vector<float>(cols / 2));
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

// Inverse 1D AWT
void inverse_awt_1d(std::vector<float> &data, const std::vector<float> &coefs, const std::vector<int> &predictors) {
    int n = coefs.size();
    
    // Inverse update step
    for (int i = 0; i < n; ++i) {
        data[2 * i] -= coefs[i] * 0.5f;  // Undo the update step
    }
    
    // Inverse predict step
    for (int i = 0; i < n; ++i) {
        int odd_idx = 2 * i + 1;
        int pred_index = predictors[i];
        const auto &filter = AWT_PREDICTORS[pred_index];
        int len = filter.size();
        int start_index = odd_idx - len / 2;
        float pred_val = 0.0f;
        
        // Calculate prediction using the same predictor that was used in forward transform
        for (int j = 0; j < len; ++j) {
            int idx = clamp(start_index + j, 0, data.size() - 1);
            if (idx % 2 == 0) {
                pred_val += filter[j] * data[idx];
            }
        }
        
        // Reconstruct odd sample
        data[odd_idx] = pred_val + coefs[i];
    }
}

// Inverse 2D AWT
void inverse_awt_2d(Matrix &img, const Matrix &h_coefs, const Matrix &v_coefs, const Matrix &d_coefs,
                    const Matrix &row_pred_map, const Matrix &col_pred_map, const Matrix &diag_pred_map) {
    int rows = img.size();
    int cols = img[0].size();
    
    // Create a temporary image with the same size as the original
    Matrix temp_img = img;
    
    // First, handle diagonal coefficients
    for (int c = 0; c < cols / 2; ++c) {
        std::vector<float> diag(rows);
        std::vector<float> diag_coef(rows / 2);
        std::vector<int> diag_predictors(rows / 2);
        
        // Setup even samples (horizontal details)
        for (int r = 0; r < rows / 2; ++r) {
            diag[2 * r] = h_coefs[r][c];  // Use horizontal details as even samples
            diag_coef[r] = d_coefs[r][c]; // Use diagonal details as coefficients
            diag_predictors[r] = static_cast<int>(diag_pred_map[r][c]);
        }
        
        // Apply inverse 1D AWT to reconstruct diagonal details
        inverse_awt_1d(diag, diag_coef, diag_predictors);
        
        // Save the reconstructed horizontal details
        for (int r = 0; r < rows; ++r) {
            temp_img[r][c] = diag[r];
        }
    }
    
    // Now, handle column-wise reconstruction
    for (int c = 0; c < cols / 2; ++c) {
        std::vector<float> col(rows);
        std::vector<float> col_coef(rows / 2);
        std::vector<int> col_predictors(rows / 2);
        
        // Setup even samples (approximation) and coefficients (vertical details)
        for (int r = 0; r < rows / 2; ++r) {
            col[2 * r] = img[r][c];        // Use approximation as even samples
            col_coef[r] = v_coefs[r][c];   // Use vertical details as coefficients
            col_predictors[r] = static_cast<int>(col_pred_map[r][c]); // Use column predictor map
        }
        
        // Apply inverse 1D AWT to reconstruct columns
        inverse_awt_1d(col, col_coef, col_predictors);
        
        // Save the reconstructed column
        for (int r = 0; r < rows; ++r) {
            temp_img[r][c] = col[r];
        }
    }
    
    // Finally, handle row-wise reconstruction
    for (int r = 0; r < rows; ++r) {
        std::vector<float> row(cols);
        std::vector<float> row_coef(cols / 2);
        std::vector<int> row_predictors(cols / 2);
        
        // Setup even samples and coefficients
        for (int c = 0; c < cols / 2; ++c) {
            row[2 * c] = temp_img[r][c];        // Use reconstructed even samples
            row_coef[c] = h_coefs[r][c];        // Use horizontal details as coefficients
            row_predictors[c] = static_cast<int>(row_pred_map[r][c]); // Use row predictor map
        }
        
        // Apply inverse 1D AWT to reconstruct rows
        inverse_awt_1d(row, row_coef, row_predictors);
        
        // Save the fully reconstructed row
        for (int c = 0; c < cols; ++c) {
            img[r][c] = row[c];
        }
    }
}

// Inverse multi-level AWT
void inverse_multi_level_awt(Matrix &img, int levels,
                            const std::vector<Matrix> &horizontal_coefs,
                            const std::vector<Matrix> &vertical_coefs,
                            const std::vector<Matrix> &diagonal_coefs,
                            const std::vector<Matrix> &row_pred_maps,
                            const std::vector<Matrix> &col_pred_maps,
                            const std::vector<Matrix> &diag_pred_maps) {
    
    // Process levels in reverse order (from deepest to shallowest)
    for (int level = levels - 1; level >= 0; --level) {
        // Calculate dimensions for this level
        int level_rows = img.size() / (1 << level);
        int level_cols = img.size() / (1 << level);
        
        // If not the first level, extract the approximation region
        Matrix level_img(level_rows / 2, std::vector<float>(level_cols / 2));
        
        // Copy the approximation region from the main image
        for (int r = 0; r < level_rows / 2; ++r) {
            for (int c = 0; c < level_cols / 2; ++c) {
                level_img[r][c] = img[r][c];
            }
        }
        
        // Create a working matrix for this level
        Matrix working_img(level_rows, std::vector<float>(level_cols));
        
        // Place the approximation coefficients in the top-left quadrant
        for (int r = 0; r < level_rows / 2; ++r) {
            for (int c = 0; c < level_cols / 2; ++c) {
                working_img[r][c] = level_img[r][c];
            }
        }
        
        // Apply inverse 2D AWT for this level
        inverse_awt_2d(working_img, 
                      horizontal_coefs[level], 
                      vertical_coefs[level], 
                      diagonal_coefs[level],
                      row_pred_maps[level], 
                      col_pred_maps[level], 
                      diag_pred_maps[level]);
        
        // Place the reconstructed coefficients back into the main image
        for (int r = 0; r < level_rows; ++r) {
            for (int c = 0; c < level_cols; ++c) {
                img[r][c] = working_img[r][c];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    std::string input_file;
    Matrix original_img;
    int levels = 1;
    float threshold = 0.0f; // Default: no thresholding

    // Read command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "i:l:t:")) != -1) {
        switch (opt) {
        case 'i':
            input_file = optarg;
            load_image_from_file(input_file, original_img);
            break;
        case 'l':
            levels = std::stoi(optarg);
            break;
        case 't':
            threshold = std::stof(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -i input_file [-l level] [-t threshold]\n";
            std::cerr << "  -i input_file : Input image file\n";
            std::cerr << "  -l level      : Transform level (default: 1)\n";
            std::cerr << "  -t threshold  : Coefficient threshold for compression (default: 0.0)\n";
            exit(EXIT_FAILURE);
        }
    }

    // check empty image or levels out of range: (1 - log2(n))
    if (empty(original_img) || levels < 1 || levels > floor(log2(original_img.size()))) {
        std::cerr << "Usage: " << argv[0] << " -i input_file [-l level] [-t threshold]\n";
        exit(EXIT_FAILURE);
    }
    
    // Construct output filenames with threshold info if applicable
    std::string transform_file, reconst_file;
    if (threshold > 0.0f) {
        transform_file = input_file + "_transform_l" + std::to_string(levels) + "_t" + std::to_string(threshold) + ".txt";
        reconst_file = input_file + "_reconst_l" + std::to_string(levels) + "_t" + std::to_string(threshold) + ".txt";
    } else {
        transform_file = input_file + "_transform_l" + std::to_string(levels) + ".txt";
        reconst_file = input_file + "_reconst_l" + std::to_string(levels) + ".txt";
    }

    // Make a copy for compression
    Matrix transformed_img = original_img;

    // Apply multi-level AWT
    std::vector<Matrix> horizontal_coefs, vertical_coefs, diagonal_coefs;
    std::vector<Matrix> row_pred_maps, col_pred_maps, diag_pred_maps;
    
    std::cout << "Applying " << levels << "-level AWT..." << std::endl;
    multi_level_awt(transformed_img, levels, horizontal_coefs, vertical_coefs, diagonal_coefs,
                    row_pred_maps, col_pred_maps, diag_pred_maps);
    
    // Calculate total coefficients and initial non-zero count
    int total_coeffs = original_img.size() * original_img.size();
    Matrix approx = extract_approximation(transformed_img, levels);
    int nonzeros_before = count_nonzeros(approx, horizontal_coefs, vertical_coefs, diagonal_coefs);
    float compression_ratio_before = static_cast<float>(total_coeffs) / nonzeros_before;
    
    std::cout << "Before thresholding: " << nonzeros_before << " non-zero coefficients out of " 
              << total_coeffs << " (" << std::fixed << std::setprecision(2) 
              << compression_ratio_before << ":1 compression ratio)" << std::endl;
    
    // Apply thresholding if specified
    if (threshold > 0.0f) {
        std::cout << "Applying coefficient threshold = " << threshold << std::endl;
        apply_thresholding(horizontal_coefs, vertical_coefs, diagonal_coefs, threshold);
        
        // Calculate non-zero count after thresholding
        int nonzeros_after = count_nonzeros(approx, horizontal_coefs, vertical_coefs, diagonal_coefs);
        float compression_ratio_after = static_cast<float>(total_coeffs) / nonzeros_after;
        float zero_percent = 100.0f * (1.0f - static_cast<float>(nonzeros_after) / total_coeffs);
        
        std::cout << "After thresholding: " << nonzeros_after << " non-zero coefficients (" 
                  << std::fixed << std::setprecision(2) << compression_ratio_after 
                  << ":1 compression ratio, " << zero_percent << "% zeros)" << std::endl;
    }

    // Save transformed image
    save_image_to_file(transform_file, transformed_img);
    std::cout << "Transformed image saved to: " << transform_file << std::endl;

    // Make a copy of the transformed image for reconstruction
    Matrix reconstructed_img = transformed_img;
        
    std::cout << "\nApplying inverse transform for reconstruction..." << std::endl;
    inverse_multi_level_awt(reconstructed_img, levels,
                           horizontal_coefs, vertical_coefs, diagonal_coefs,
                           row_pred_maps, col_pred_maps, diag_pred_maps);
    
    // Save reconstructed image
    save_image_to_file(reconst_file, reconstructed_img);
    std::cout << "Reconstructed image saved to: " << reconst_file << std::endl;
    
    // Compute metrics between original and reconstructed image
    float recon_mse = compute_mse(original_img, reconstructed_img);
    float recon_psnr = 10 * log10(255 * 255 / (recon_mse > 0 ? recon_mse : 1e-10));
    float recon_ssim = compute_ssim(original_img, reconstructed_img);
    
    std::cout << "\nReconstruction Quality:" << std::endl;
    std::cout << "MSE:  " << std::fixed << std::setprecision(6) << recon_mse << std::endl;
    std::cout << "PSNR: " << std::fixed << std::setprecision(2) << recon_psnr << " dB" << std::endl;
    std::cout << "SSIM: " << std::fixed << std::setprecision(6) << recon_ssim << std::endl;
    
    return 0;
}