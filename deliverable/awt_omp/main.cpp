#include "awt_shared.hpp"
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

int main(int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();
    std::string input_file;
    Matrix original_img;
    int levels = 1;
    double threshold = 0.0;
    int num_threads = 1;

    // Read command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "i:l:t:n:")) != -1) {
        switch (opt) {
        case 'i':
            input_file = optarg;
            break;
        case 'l':
            levels = std::stoi(optarg);
            break;
        case 't':
            threshold = std::stof(optarg);
            break;
        case 'n':
            num_threads = atoi(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -i input_file [-l level] [-t threshold]\n";
            exit(EXIT_FAILURE);
        }
    }

    // check params
    if (threshold < 0.0f || num_threads <= 0) {
        std::cerr << "Usage: " << argv[0] << " -i input_file [-l level] [-t threshold]\n";
        exit(EXIT_FAILURE);
    }

    load_image_from_file(input_file, original_img);

    // check image or levels out of range: (1 - log2(n))
    if (empty(original_img) || levels < 1 || levels > floor(log2(original_img.size()))) {
        std::cerr << "Usage: " << argv[0] << " -i input_file [-l level] [-t threshold]\n";
        exit(EXIT_FAILURE);
    }

    omp_set_num_threads(num_threads);

    std::string filename = extract_base_name(input_file);

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    const auto transform_start = std::chrono::steady_clock::now();

    std::vector<Matrix> row_pred_maps, col_pred_maps, diag_pred_maps;
    Matrix transformed_img = original_img;
    std::string transformed_file = "../dataset/transformed/" + filename +
                                   "_level_" + std::to_string(levels) +
                                   "_t_" + std::format("{:.2f}", threshold) + ".txt";

    std::chrono::steady_clock::time_point reconst_start;
    std::string reconst_file = "../dataset/reconstructed/" + filename +
                               "_level_" + std::to_string(levels) +
                               "_t_" + std::format("{:.2f}", threshold) + ".txt";
    Matrix reconst_img;

    // Shared Memory mode
    std::cout << "Running in Shared Memory Model...\n";

    // Apply multi-level AWT
    awt_multi_level_shared(transformed_img, levels, threshold, row_pred_maps, col_pred_maps, diag_pred_maps);

    // TODO: visulizing the coefficients

    reconst_start = std::chrono::steady_clock::now();

    reconst_img = transformed_img;
    reconst_awt_shared(reconst_img, levels,
                       row_pred_maps, col_pred_maps, diag_pred_maps);

    // const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - transform_start).count();
    // std::cout << "\033[31mComputation time (sec): " << std::fixed << std::setprecision(10) << compute_time << "\033[0m\n";
    const double transformation_time = std::chrono::duration_cast<std::chrono::duration<double>>(reconst_start - transform_start).count();
    std::cout << "\033[31mTransformation time (sec): " << std::fixed << std::setprecision(10) << transformation_time << "\033[0m\n";
    const double reconstruction_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - reconst_start).count();
    std::cout << "\033[34mReconstruction time (sec): " << std::fixed << std::setprecision(10) << reconstruction_time << "\033[0m\n";
    // const double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    // std::cout << "\033[34mTotal time (sec): " << std::fixed << std::setprecision(10) << total_time << "\033[0m\n";

    // Save transformed image
    // save_image_to_file(transformed_file, transformed_img);
    // Save reconstructed image
    // save_image_to_file(reconst_file, reconst_img);
    std::cout << "Reconstructed image saved to: " << reconst_file << std::endl;

    // Metrics computation
    float recon_mse = compute_mse(original_img, reconst_img);
    float recon_ssim = compute_ssim(original_img, reconst_img);

    std::cout << "MSE: " << recon_mse << std::endl;
    std::cout << "SSIM: " << recon_ssim << std::endl;

    return 0;
}