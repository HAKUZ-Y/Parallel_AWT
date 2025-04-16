# Parallel 2D Multi-level Adaptive Wavelet Transformation

Maggie Gong & Youheng Yang

## URL: 
- Proposal: https://hakuz-y.github.io/Parallel_AWT/proposal/
- Milestone: https://hakuz-y.github.io/Parallel_AWT/milestone/

## Summary

This project aims to implement a parallel version of the Adaptive Wavelet Transformation inspired by the [paper](https://apps.dtic.mil/sti/pdfs/ADA372394.pdf) with both shared memory model and MPI. AWT dynamically selects wavelet bases tailored to local image features, aiming to improve compression and rendering performance over traditional fixed-basis wavelet transforms. The primary objective is to enhance computational performance while preserving scalability and output quality. This involves addressing the unique challenges associated with parallelizing adaptive algorithms, such as load imbalance, data dependencies, and irregular memory access patterns.


## Schedule
- Week 1 (Mar 26th): complete the proposal, brainstorm idea, finalize project outline ✅
- Week 2 (April 2nd): implement the sequential C++ version of the algorithm, create test cases ✅
- Week 3 (April 15th): initial parallel implementation, create more comprehensive test cases, milestone report ✅
- By April 18th, Fri: Having an idea and design of MPI (Youheng), further optimize the algorithm (Maggie).
- By April 20th, Sun: Having an implementation of MPI (Youheng & Maggie).
- By April 25th, Fri: 
    - Optimize MPI to achieve the targeting speedup (4-5x with 8 threads) (Youheng & Maggie). 
    - Experiment with PSC if we have the access (Youheng & Maggie).
    - If the above is done by 20th, try to optimize both version with a perfect speedup (10-11x with 12 threads) (Youheng & Maggie).
    - If we were really lucky and achieve all of the above early, try to implement the lock-free version (Youheng & Maggie).
- By April 27th, Sun:
    - Final report.
    - Poster Sessions preparation.
- By April 28th, Mon: Final review on the report and poster session presentation.