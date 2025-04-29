#!/bin/bash

mpic++ -std=c++17 -fopenmp -o awt_mpi awt_mpi.cpp helper.cpp

if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

thread_counts=(1 2 4 8)
image_files=("easy_2048" "medium_8192" "hard_14992")

for image_file in "${image_files[@]}"; do
    for thread_num in "${thread_counts[@]}"; do
        echo "========================================="
        echo "Running with $thread_num threads on $image_file"
        mpirun -np "$thread_num" ./awt_mpi "$image_file"
    done
done

echo "All runs completed."
