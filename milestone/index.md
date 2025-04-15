# Milestone Report


## Schedule
- ~~Week 1 (Mar 26th): complete the proposal, brainstorm idea, finalize project outline~~
- ~~Week 2 (April 2nd): implement the sequential C++ version of the algorithm, create test cases~~
- ~~Week 3 (April 15th): initial parallel implementation, create more comprehensive test cases, milestone report~~
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



## Completed

For the first 2 weeks, we have completed the Open_MP version of parallelizing the 2D multi-level Adaptive Wavelet Transformation that processes image compression. We learned the algorithm from the paper [Adaptive Wavelet Transforms
via Lifting](https://apps.dtic.mil/sti/tr/pdf/ADA372394.pdf) and implemented a baseline from it. Then, we used Open_MP to parallelize it in a shared memory model. We examed the initialization time, transformation time, reconstruction time and achieved a 4-5x speed up with 12 threads. We also created the metrics with MSE (Mean Squared Error) and SSIM (Structural similarity index measure) to evaluate the quality of the compressed image. The detailed results are in the section `Preliminary results`.

TODO: some details on the algo and parallel impl

## Goals and Deliverables

Currently, we followed our schedule and goals well in the proposal. In the first week, we successfully implemented a baseline of image compression by using a 2D multi-level Adaptive Wavelet Transformation and Reconstruction with the following steps: 1D [discrete wavelet transformation](https://en.wikipedia.org/wiki/Discrete_wavelet_transform) --> 2D DWT --> 2D multi-level DWT -> 2D multi-level AWT.
In the second week, we created 3 test dataset: easy_2048, medium_8192, hard_16384. Since the largest test file has a size of 1.6GB, we didn't push it into our repo, but included the original image and the script to generate the grayscale txt. We also created 2 metrics, MSE and ISSM to test the correctness of our implementation, which should be 0 and 1 respectively for a threshold of 0. Finally, we parallelized the transformation and reconstruction by using the Open_MP, acheving a 4-5x speedup with 12 threads.

We believe we could finish the next deliverable that uses MPI model. Though we are not fully confident to achieve an optimal speedup as the shared memory model due to the communication complexity of AWT, we will try to optimize it. similarly, the perfect speedup in the "nice to haves" section of the proposal might be hard to achieve, but we will use different strategies to optimize it as much as possible.


## Plan for poster session

- Visualizations:
    - A simple demo compares the image compression speed between a sequential, shared memory model, and MPI model.
    - Some plots shows the speedup and metrics of 2 different parallelism with different levels and threshold

- Graphs:
    - Some main conclusion in our projects.
    - What we learned from our sources, and what we learned during the implementation and optimization.
    - Some challenges and limitations we met during the implementation.


## Preliminary results

TODO: speedup plots


## Concerns

With the complexity of the algorithm and parallelizing the AWT, we are not very confident to achieve a similar speed up with MPI version, as there would be a lots of design and communication overhead. We will try different orchestrations, and some possible papers or dicussions in public to see if we could find some good intuition behind it.

Another difficulty is the size of the test file, which is about 1.6 GB for the hard case. We were able to test it with at most 12 threads in our local computers, but we are not sure if we have access to the PSC, and if we have enough volume for the test files.