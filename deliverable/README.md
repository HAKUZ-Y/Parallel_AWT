# Parallelize 2D Multi-level Adaptive Wavelet Transformation

## awt_omp:
This is the source code for omp version of awt, to execute it
- `./awt -i <path/to/dataset> -l <level, default=1> -n <thread_num, default=1> -t <threshold, default=0>`
- Example: `./awt -i ../dataset/grayscale/easy_2048.txt -l 1 -n 12`

## awt_mpi:
To compile:
- Ensure all the tools needed are installed, else `brew install openmpi`
- 1D AWT: `mpic++ -std=c++17 -fopenmp -o awt_1d_mpi awt_1d_mpi.cpp`
- 2D AWT: `mpic++ -std=c++17 -fopenmp -o awt_mpi awt_mpi.cpp helper.cpp`

To execute:
- 1D AWT: `mpirun -np [number of processes] ./awt_1d_mpi`
  > Note that the test case was written inside the `awt_1d_mpi.cpp` file, so if you wish to modify it, you need to edit the source file and recompile.
- 2D AWT: `mpirun -np [number of processes] ./awt_mpi [file name (without .png)]`
  > Example: `mpirun -np 12 ./awt_mpi hard_14992`

## running on PSC machines:
- `interact -N [number of nodes] -n 128 --partition RM`
- `module purge`
- `module load openmpi cuda nvhpc`
- Enter interactive shell

## assumptions:
- The input image is squared.
- The level is between 1 and log_2(data_size).
- The threshold should be non-negative.


## Dataset:
https://github.com/HAKUZ-Y/Parallel_AWT/blob/main/deliverable/dataset/README.md
