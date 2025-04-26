#include "awt_mpi.hpp"
#include "awt_shared.hpp"
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

int main(int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();

    int pid;
    int nproc;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    std::string input_file;
    Matrix original_img;
    int levels = 1;
    double threshold = 0.0;

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
        default:
            std::cerr << "Usage: " << argv[0] << " -i input_file [-l level] [-t threshold]\n";
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    // check params
    if (threshold < 0.0f) {
        std::cerr << "Usage: " << argv[0] << " -i input_file [-l level] [-t threshold]\n";
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    std::string transformed_file;
    std::string reconst_file;
    std::chrono::steady_clock::time_point transform_start;
    std::chrono::steady_clock::time_point reconst_start;
    std::vector<Matrix> row_pred_maps, col_pred_maps, diag_pred_maps;

    if (pid == 0) {
        std::cout << "Running in MPI Model...\n";

        load_image_from_file(input_file, original_img);

        // check image or levels out of range: (1 - log2(n))
        if (empty(original_img) || levels < 1 || levels > floor(log2(original_img.size()))) {
            std::cerr << "Usage: " << argv[0] << " -i input_file [-l level] [-t threshold] [-m mode]\n";
            exit(EXIT_FAILURE);
        }

        std::string filename = extract_base_name(input_file);

        const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
        std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

        transformed_file = "../dataset/transformed/" + filename +
                           "_level_" + std::to_string(levels) +
                           "_t_" + std::format("{:.2f}", threshold) + ".txt";

        reconst_file = "../dataset/reconstructed/" + filename +
                       "_level_" + std::to_string(levels) +
                       "_t_" + std::format("{:.2f}", threshold) + ".txt";
    }

    int rows, cols;

    if (pid == 0) {
        rows = original_img.size();
        cols = original_img[0].size();
    }

    // Broadcast dimensions
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> buffer(rows * cols);

    if (pid == 0) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                buffer[r * cols + c] = original_img[r][c];
            }
        }
    }

    // Broadcast buffer
    MPI_Bcast(buffer.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (pid != 0) {
        original_img.resize(rows, std::vector<double>(cols));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                original_img[r][c] = buffer[r * cols + c];
            }
        }
    }

    if (pid == 0) {
        transform_start = std::chrono::steady_clock::now();
    }
    Matrix transformed_img = original_img;
    awt_multi_level_mpi(transformed_img, levels, threshold, row_pred_maps, col_pred_maps, diag_pred_maps, pid, nproc);

    if (pid == 0) {
        reconst_start = std::chrono::steady_clock::now();
    }
    Matrix reconst_img = transformed_img;
    reconst_awt_mpi(reconst_img, levels,
                    row_pred_maps, col_pred_maps, diag_pred_maps);

    if (pid == 0) {
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
    }

    MPI_Finalize();
    return 0;
}