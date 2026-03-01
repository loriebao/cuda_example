# cuda_example
# A GPU is composed of multiple SMs (the hardware). When you launch a kernel, your Grid is distributed across these SMs.
# Grid: The entire collection of threads for a kernel launch.
# Block: A group of threads within the Grid. One Block is the smallest unit of work assigned to an SM. Once a block is assigned to an SM, it stays there until all its threads finish.
# SM (Streaming Multiprocessor): The actual hardware unit. An SM can run multiple blocks concurrently, but a single block cannot be split across multiple SMs
# The maximum number of concurrent warps per SM is 64 for compute capability 10.0
# The register file size is 64K 32-bit registers per SM.
# The maximum number of registers per thread is 255.
# The maximum number of thread blocks per SM is 32 for devices of compute capability 10.0 and 12.0.
# For devices of compute capability 10.0 shared memory capacity per SM is 228 KB. 
# For devices of compute capability 10.0 the maximum shared memory per thread block is 227 KB.
# For applications using Thread Block Clusters, it is always recommended to compute the occupancy using cudaOccupancyMaxActiveClusters and launch cluster-based kernels accordingly.
nvidia-smi --query-gpu=compute_cap --format=csv (A4000: 8.6, B4000: 10.x)
