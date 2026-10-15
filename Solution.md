# GPUs vs CPUs (lab 1: architecture and performance)
The goal of this lab is to develop a deeper understanding of the fundamental architectural differences between GPUs and CPUs and to observe how these differences impact performance.

## Tasks (system info)

1. GPU architecture: (NVIDIA RTX PRO 4000) Blackwell 
2. No. SMs: 70  
3. No. SPs: 8960 (Cuda)
4. CPU: Intel(R) Xeon(R) w3-2525    
5. No. threads CPU vs no. threads GPU: 16 vs (70 x 8960) 143360


Source GPU: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/workstation-datasheet-blackwell-rtx-pro-4000-nvidia-3662515.pdf
            https://www.techpowerup.com/gpu-specs/rtx-pro-4000-blackwell-workstation.c4279

Source CPU: https://www.intel.com/content/www/us/en/products/sku/236882/intel-xeon-w32525-processor-22-5m-cache-3-50-ghz/specifications.html


## Tasks (performance)
1.  Vector addition of 50000 elements
    CUDA kernel launch with 196 blocks of 256 threads
    Test PASSED

2.  Vector addition of 50000 elements
    CUDA kernel launch with 196 blocks of 256 threads
    Test PASSED (GPU and CPU results match)
    GPU Execution Time: 0.155904 ms

3. addition.c vectorAddCPU

4. Vector addition of 50000 elements
    CUDA kernel launch with 196 blocks of 256 threads
    Test PASSED (GPU and CPU results match)
    CPU Execution Time: 0.177000 ms

5. 
    256 blocksize:

    Vector addition of 500000 elements:
    GPU Execution Time: 0.266848 ms
    CPU Execution Time: 1.743000 ms

    Vector addition of 5000000 elements:
    GPU Execution Time: 0.360448 ms
    CPU Execution Time: 17.070000 ms

    Vector addition of 50000000 elements
    GPU Execution Time: 1.338240 ms
    CPU Execution Time: 137.231000 ms

    Vector addition of 500000000 elements
    GPU Execution Time: 11.733536 ms
    CPU Execution Time: 1367.488000 ms

    512 blocksize:

Vector addition of 50000 elements
CUDA kernel launch with 98 blocks of 512 threads
Test PASSED (GPU and CPU results match)
GPU Execution Time: 0.203808 ms
CPU Execution Time: 0.196000 ms

Vector addition of 500000 elements
CUDA kernel launch with 977 blocks of 512 threads
Test PASSED (GPU and CPU results match)
GPU Execution Time: 0.264992 ms
CPU Execution Time: 1.723000 ms

Vector addition of 5000000 elements
CUDA kernel launch with 9766 blocks of 512 threads
Test PASSED (GPU and CPU results match)
GPU Execution Time: 0.342912 ms
CPU Execution Time: 17.157000 ms

Vector addition of 50000000 elements
CUDA kernel launch with 97657 blocks of 512 threads
Test PASSED (GPU and CPU results match)
GPU Execution Time: 1.280032 ms
CPU Execution Time: 170.855000 ms

Vector addition of 500000000 elements
CUDA kernel launch with 976563 blocks of 512 threads
Test PASSED (GPU and CPU results match)
GPU Execution Time: 11.901216 ms
CPU Execution Time: 1359.807000 ms

1024 Blocksize:

Vector addition of 50000 elements
CUDA kernel launch with 49 blocks of 1024 threads
Test PASSED (GPU and CPU results match)
GPU Execution Time: 0.171808 ms
CPU Execution Time: 0.174000 ms

Vector addition of 500000 elements
CUDA kernel launch with 489 blocks of 1024 threads
Test PASSED (GPU and CPU results match)
GPU Execution Time: 0.247040 ms
CPU Execution Time: 1.660000 ms

Vector addition of 5000000 elements
CUDA kernel launch with 4883 blocks of 1024 threads
Test PASSED (GPU and CPU results match)
GPU Execution Time: 0.346400 ms
CPU Execution Time: 16.492000 ms

Vector addition of 50000000 elements
CUDA kernel launch with 48829 blocks of 1024 threads
Test PASSED (GPU and CPU results match)
GPU Execution Time: 1.301728 ms
CPU Execution Time: 169.867000 ms

Vector addition of 500000000 elements
CUDA kernel launch with 488282 blocks of 1024 threads
Test PASSED (GPU and CPU results match)
GPU Execution Time: 11.765472 ms
CPU Execution Time: 1363.361000 ms




6. plot_cpu_vs_gpu.png


7. Compile and run race condition exercise

8. Pass the race condition test by identifying the bug in the _countAboveThreshold_ CUDA kernel and solving it.

inside countAboveThreshold():


if (i < numElements && A[i] > threshold) {
        //(*count)++;
        atomicAdd(count, 1);

}

You have to replace the (*count)++ incrementation with the atomicAdd(count, 1).
Otherwise the part of the memory that increments the value will be overwritten, this way the incerement correctly works, but atomicAdd will significantly slow down the execution time if we scale the number of elements up.
## Wrap up and deliverables
1. Create blog post (or GitHub repo)
2. Describe main applications for GPUs (graphics and AI)
3. Showcase your experiments and speed ups achieved
4. Segway to architectural differences (describe CPU vs GPU during lab, cores, threads, and other specifics)
5. Showcase speed up experiments and describe differences (especially for today)
6. Discuss the race condition problem, how they might appear, and how they are usually treated.
7. Wrap up and conclude with what's next

