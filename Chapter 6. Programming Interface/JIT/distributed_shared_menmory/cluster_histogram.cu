#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block,
                                   const int *__restrict__ input, size_t array_size) {
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;

    int tid = cg::this_grid().thread_rank();
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
        smem[i] = 0;
    }
    cluster.sync();

    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
        int ldata = input[i];
        int binid = ldata;
        if (ldata < 0) binid = 0;
        else if (ldata >= nbins) binid = nbins - 1;

        int dst_block_rank = (int)(binid / bins_per_block);
        int dst_offset = binid % bins_per_block;
        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
        atomicAdd(dst_smem + dst_offset, 1);
    }
    cluster.sync();

    int *lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
        atomicAdd(&lbins[i], smem[i]);
    }
}

int main() {
    const int NBINS = 64;
    const int BLOCK_SIZE = 256;
    const int BINS_PER_BLOCK = 16;
    const size_t ARRAY_SIZE = 1024;
    int cluster_size = NBINS / BINS_PER_BLOCK;

    int *h_input = (int*)malloc(ARRAY_SIZE * sizeof(int));
    int *h_bins = (int*)malloc(NBINS * sizeof(int));

    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = rand() % NBINS;
    }
    for (int i = 0; i < NBINS; i++) {
        h_bins[i] = 0;
    }

    int *d_input, *d_bins;
    cudaMalloc(&d_input, ARRAY_SIZE * sizeof(int));
    cudaMalloc(&d_bins, NBINS * sizeof(int));

    cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, h_bins, NBINS * sizeof(int), cudaMemcpyHostToDevice);

    // 为按值传递的参数创建副本
    int nbins = NBINS;
    int bins_per_block = BINS_PER_BLOCK;
    size_t array_size = ARRAY_SIZE;

    // 传递参数的指针
    void *kernel_args[] = {
        &d_bins,          // int*
        &nbins,           // const int
        &bins_per_block,  // const int
        &d_input,         // const int*
        &array_size       // size_t
    };

    cudaLaunchCooperativeKernel((void*)clusterHist_kernel,
                                cluster_size,
                                BLOCK_SIZE,
                                kernel_args,
                                BINS_PER_BLOCK * sizeof(int),
                                0);

    cudaMemcpy(h_bins, d_bins, NBINS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Histogram results (first 10 bins):\n");
    for (int i = 0; i < 10; i++) {
        printf("Bin %d: %d\n", i, h_bins[i]);
    }

    cudaFree(d_input);
    cudaFree(d_bins);
    free(h_input);
    free(h_bins);

    return 0;
}