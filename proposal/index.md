# Parallel 2D Multi-level Adaptive Wavelet Transformation

Maggie Gong & Youheng Yang

## URL: https://hakuz-y.github.io/Parallel_AWT/proposal/

## Summary:

This project aims to implement a parallel version of the Adaptive Wavelet Transformation inspired by the [paper](https://apps.dtic.mil/sti/pdfs/ADA372394.pdf) with both shared memory model and MPI. AWT dynamically selects wavelet bases tailored to local image features, aiming to improve compression and rendering performance over traditional fixed-basis wavelet transforms. Our goal is to enhance performance while maintaining a good scalability and quality for image processing such as compression and rendering, addressing the challenges associated with parallelizing adaptive algorithms.

## Background:

Wavelet transforms are used for analyzing data where features vary over different scales in signal and image processing for tasks such as compression, denoising, and multi-resolution analysis. Real-world signals often have smooth regions interrupted by abrupt changes, so rapidly decaying wave-like oscillation are being used to represent such data.

To make it more interesting in parallelizing, AWT adapts wavelet basis functions based on local image characteristics, introducing significant data dependencies. The computation in one region can influence decisions in adjacent areas, complicating parallel execution.

Adaptive Wavelet Transform Overview:
- Image Analysis: The image is divided into blocks, and each block is analyzed to identify characteristics such as edges, textures, and smooth regions.​
- Basis Function Selection: Based on the analysis, the most suitable wavelet basis function is selected for each block to best represent its features.​
- Transformation: The selected wavelet transform is applied to each block, resulting in coefficients that are more efficiently compressed due to their alignment with local image properties.

In addition to parallelism, we also plan to explore various strategies for memory access, including in-place and out-of-place implementations; with the former one having less memory overhead but higher complexity and the latter one having more straightforward parallelization but causing extra memory costs.


## The Challenge:

Workload Imbalance:
- The main challenge for parallelizing adaptive wavelet transform is the load balancing for highly inhomogeneous problems with spatially non-uniform distribution of data points at each level of resolution, since the algorithms developed for non-adaptive wavelet transform fail, mainly because the parallel non-adaptive algorithms require synchronization of each stage of wavelet transform on each level of resolution, which is very impractical from either data locality or load balancing standpoint. 

Data Dependencies:
- As wavelet coefficients overlap spatially, a single sample at location x affects multiple coefficients across levels due to the projection into wavelet basis, causing write conflicts when parallel threads try to update overlapping coefficients.
- In the 2D Multi-level AWT, each wavelet transformation level depends on the previous level's coefficients, and there is also a sequential dependency where level N cannot start until level N-1 is complete, and we need to work around this potential bottleneck between levels.

Memory access characteristics
- Memory access is irregular, so there’s non-contiguous memory writes during coefficient generation. The irregular data access pattern could cause cache utilization to drop.
- Non-contiguous memory writes during coefficient generation

Data Movements:
- There is significant data shuffling between levels during the computation, and this is a multi-level 2-dimensional AWT.


## Resources:

[Adaptive Wavelet Transformation](https://www.cosy.sbg.ac.at/~rkutil/publication/Kutil00a.pdf) \
[Adaptive Wavelet Rendering](https://cseweb.ucsd.edu/~ravir/Overbeck2009AWR.pdf) \
It would be really helpful if we could have access of PSC machines and test beyond 8 threads.


## Goals and Deliverables:

Plan to achieve:

1. Develop a fully functional sequential implementation of AWT application (Image Compression/Rendering).
2. Parallelize the sequential implementation using shared memory model.
3. Based on step 2, improve the parallel version’s scalability by using OpenMP.
4. Establish benchmarks (both speedups and scalability) to measure performances.
5. Profile and analyze the performances of step 1, 2, and 3.
6. Detailed documentation of our approach and progress.
7. Achieve a certain level of speedups (4 to 5x with 8 threads).

Hope to achieve:

1. Achieve an ideal speedup (7 to 8x with 8 threads).
2. We could also explore transactional memory v.s. locking v.s. lock-free techniques for synchronizing access to shared resources, as they have an effect on both performance and scalability.

We hope by the end of this project, we develop a deeper understanding of the challenges in parallelizing a highly complex sequential algorithm , as well as the key roles that memory access pattern have in affecting performances of a parallel program.

## Platform Choice:

- For CPU parallelization, we plan to use GHC and PSC machines for testing, with number of processes ranging from 1 to 128. We will start with implementing the parallel version using OpenMP with shared memory model for ease of development and then transition to MPI to improve scalability across larger number of processes.
- We will use C++ as our programming language given its performance for efficient low-level control and suitability

## Schedule:

- Week 1 (Mar 26th): complete the proposal, brainstorm idea, finalize project outline
- Week 2 (April 2nd): implement the sequential C++ version of the algorithm, create test cases
- Week 3 (April 15th): initial parallel implementation, create more comprehensive test cases
- Week 4 (April 22nd): milestone report, transition from shared memory model to MPI, profiling
- Week 5 (April 28th): more profiling, data analysis and visualization, improvements, prepare final report
- Week 5 (April 29th): presentation