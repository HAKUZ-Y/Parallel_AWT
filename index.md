# Parallel 2D Multi-level Adaptive Wavelet Transformation

Maggie Gong & Youheng Yang

## URL: 
- [Proposal Page](https://hakuz-y.github.io/Parallel_AWT/proposal/)
- [Milestone Page](https://hakuz-y.github.io/Parallel_AWT/milestone/)

## Summary

This project aims to implement a parallel version of the Adaptive Wavelet Transformation inspired by the [paper](https://apps.dtic.mil/sti/pdfs/ADA372394.pdf) with both shared memory model and MPI. AWT dynamically selects wavelet bases tailored to local image features, aiming to improve compression and rendering performance over traditional fixed-basis wavelet transforms. The primary objective is to enhance computational performance while preserving scalability and output quality. This involves addressing the unique challenges associated with parallelizing adaptive algorithms, such as load imbalance, data dependencies, and irregular memory access patterns.


## Schedule
- Week 1 (Mar 26th): complete the proposal, brainstorm idea, finalize project outline ✅
- Week 2 (April 2nd): implement the sequential C++ version of the algorithm, create test cases ✅
- Week 3 (April 15th): initial parallel implementation, create more comprehensive test cases, milestone report ✅

| Deadline | Task | Partner |
|------|------|---------|
|4/16| Improve the sequential implementation (inplace details) | Maggie |
|4/17| Optimize the shared memory model | Youheng |
|4/18| Further optimize the algorithm | Maggie |
|4/19| Finish current TODOs in the baseline and shared memory version |Youheng|
|4/20| Transition to MPI implementation| Together |
|4/21| Optimize MPI to achieve the targeting speedup (4-5x with 8 threads) | Maggie |
|4/22| Experiment with PSC if we have the access, achieve the targeting speedup | Youheng |
|4/23| If the above is done by 20th, try to optimize both version with a perfect speedup (10-11x with 12 threads)| Together |
|4/24| If we were really lucky and achieve all of the above early, try to implement the lock-free version | Together |
|4/25| Profiling, Collecting speedup graphs | Together |
|4/27| draft the final report, poster session preparation | Together |
|4/28| Wrap up, final review on the report and poster session presentation | Together |
|4/29| Presentation | Together |
