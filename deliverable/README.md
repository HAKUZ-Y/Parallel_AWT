# Parallelize 2D Multi-level Adaptive Wavelet Transformation

## awt_omp:
This is the source code for omp version of awt, to execute it
- `./awt -i <path/to/dataset> -l <level, default=1> -n <thread_num, default=1> -t <threshold, default=0>`
- Example: `./awt -i ../dataset/grayscale/easy_2048.txt -l 1 -n 12`

## awt_mpi:
This is the source code for mpi version of awt, to execute it
- `./awt -i <path/to/dataset> -l <level, default=1> -t <threshold, default=0>`
- Example: `mpirun -np 12 ./awt -i ../dataset/grayscale/easy_2048.txt -l 1`

## assumptions:
- The input image is squared.
- The level is between 1 and log_2(data_size).
- The threshold should be non-negative.


## Dataset:
https://github.com/HAKUZ-Y/Parallel_AWT/blob/main/deliverable/dataset/README.md
