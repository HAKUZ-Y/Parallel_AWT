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

/*******************************************************************************
 *                               AWT Transformation                            *
 *******************************************************************************/
void awt_transform_1d_mpi(std::vector<float> &data, std::vector<float> &coefs, std::vector<float> &predictors, int pid, int nproc) {
    int total_n = data.size(); // [approx | detail]
    int n = total_n / 2;
    coefs.resize(n);
    predictors.resize(n);

    int chunk_size = n / nproc;
    int remainder = n % nproc;

    // distribute the input signals among the processors (considering remainders)
    std::vector<int> counts(nproc), displs(nproc);
    for (int i = 0; i < nproc; ++i) {
        counts[i] = chunk_size + (remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    int local_n = counts[pid]; // NOTE: these r all local to the processor
    int start_idx = displs[pid];
    int end_idx = start_idx + local_n;

    // local buffers
    std::vector<float> local_coefs(local_n);
    std::vector<float> local_preds(local_n);

    // processing the assigned chunk of data
    for (int i = 0; i < local_n; ++i) {
        int global_idx = start_idx + i;
        int odd_idx = 2 * global_idx + 1;
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
                int idx = clamp(start_index + j, 0, total_n - 1);
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
        local_coefs[i] = min_coef;
        local_preds[i] = static_cast<float>(pred_index);
    }
    // Update step (assumes symmetric filters), adaptive

    // TODO try async communication
    // using allgatherv instead of allgather to account for the remainders
    // https: // rookiehpc.org/mpi/docs/mpi_allgatherv/index.html
    MPI_Allgatherv(local_coefs.data(), local_n, MPI_FLOAT,
                   coefs.data(), counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(local_preds.data(), local_n, MPI_FLOAT,
                   predictors.data(), counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

    // update
    std::vector<float> result(total_n); // NOTE: this is also local to the proc
    for (int i = 0; i < local_n; ++i) {
        int global_i = start_idx + i;
        result[2 * global_i] = data[2 * global_i];

        const auto &update_filter = AWT_UPDATES[predictors[global_i]];
        int update_filter_len = update_filter.size();
        int start = global_i - update_filter_len / 2;

        float update_sum = 0.0f;

        for (int j = 0; j < update_filter_len; ++j) {
            int idx = clamp(start + j, 0, n - 1); // TODO: padding for edges
            update_sum += update_filter[j] * coefs[idx];
        }

        result[2 * global_i] += update_sum;         // update approx coefficient
        result[2 * global_i + 1] = coefs[global_i]; // detail in the second half
    }

    // gather all results (many-to-many comm)
    // TODO make this async
    std::vector<int> counts_2n(nproc), displs_2n(nproc);
    for (int i = 0; i < nproc; ++i) {
        counts_2n[i] = counts[i] * 2;
        displs_2n[i] = (i == 0) ? 0 : displs_2n[i - 1] + counts_2n[i - 1];
    }

    MPI_Allgatherv(result.data() + 2 * start_idx, local_n * 2, MPI_FLOAT,
                   data.data(), counts_2n.data(), displs_2n.data(), MPI_FLOAT, MPI_COMM_WORLD);
}

void test(std::vector<float> signal, int pid, int nproc) {
    std::vector<float> original = signal;
    std::vector<float> coefs;
    std::vector<float> predictors;

    // if (pid == 0) {
    //     std::cout << "Original:\n";
    //     for (float x : signal)
    //         std::cout << x << " ";
    //     std::cout << "\n";
    // }

    MPI_Bcast(signal.data(), signal.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    const auto compute_start = std::chrono::steady_clock::now();

    // transformation
    awt_transform_1d_mpi(signal, coefs, predictors, pid, nproc);

    if (pid == 0) {
        // std::cout << "After AWT ([approx | detail]):\n";
        // for (float x : signal)
        //     std::cout << x << " ";
        // std::cout << "\n";

        const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
        std::cout << "\033[31mComputation time (sec): " << std::fixed << std::setprecision(10) << compute_time << "\033[0m\n";
    }
}

std::vector<float> generate_large_signal(int size) {
    std::vector<float> signal(size);
    for (int i = 0; i < size; ++i) {
        signal[i] = i + i / 2; // repeating pattern with offset
    }
    return signal;
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

    // std::vector<float> signal = {1.1, 2.2, 3.3, 4.4, 5.5, 6.66, 7.77, 8.88, 9.99, 10};
    std::vector<float> signal = generate_large_signal(1000); // 1 million elements

    test(signal, pid, nproc);

    MPI_Finalize();
    return 0;
}
