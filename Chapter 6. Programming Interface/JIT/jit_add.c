#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// 检查 CUDA API 调用是否成功
#define CHECK_CUDA(call) { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", errStr, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define N 4

int main() {
    // 初始化 CUDA 驱动 API
    CHECK_CUDA(cuInit(0));

    // 获取第一个 CUDA 设备
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    // 创建 CUDA 上下文
    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    // 加载 PTX 文件
    CUmodule module;
    CHECK_CUDA(cuModuleLoad(&module, "navie_add.ptx")); // 假设 PTX 文件名为 "add.ptx"

    // 获取核函数
    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "_Z3addPiS_S_"));

    // 准备主机端数据
    int h_a[N] = {1, 2, 3, 4};
    int h_b[N] = {5, 6, 7, 8};
    int h_c[N] = {0};

    // 在设备上分配内存
    CUdeviceptr d_a, d_b, d_c;
    CHECK_CUDA(cuMemAlloc(&d_a, N * sizeof(int)));
    CHECK_CUDA(cuMemAlloc(&d_b, N * sizeof(int)));
    CHECK_CUDA(cuMemAlloc(&d_c, N * sizeof(int)));

    // 将数据从主机复制到设备
    CHECK_CUDA(cuMemcpyHtoD(d_a, h_a, N * sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(d_b, h_b, N * sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(d_c, h_c, N * sizeof(int)));

    // 设置核函数参数
    void *args[] = { &d_a, &d_b, &d_c };

    // 配置线程块和网格
    int threadsPerBlock = N; // 每个线程处理一个元素
    int blocksPerGrid = 1;   // 简单起见，使用一个块

    // 启动核函数
    CHECK_CUDA(cuLaunchKernel(kernel,
                              blocksPerGrid, 1, 1,    // 网格维度 (x, y, z)
                              threadsPerBlock, 1, 1,  // 块维度 (x, y, z)
                              0, NULL,                // 共享内存和流
                              args, NULL));           // 参数和额外选项

    // 将结果从设备复制回主机
    CHECK_CUDA(cuMemcpyDtoH(h_c, d_c, N * sizeof(int)));

    // 打印结果
    printf("Results:\n");
    for (int i = 0; i < N; i++) {
        printf("c[%d] = %d\n", i, h_c[i]);
    }

    // 清理资源
    CHECK_CUDA(cuMemFree(d_a));
    CHECK_CUDA(cuMemFree(d_b));
    CHECK_CUDA(cuMemFree(d_c));
    CHECK_CUDA(cuModuleUnload(module));
    CHECK_CUDA(cuCtxDestroy(context));

    return 0;
}