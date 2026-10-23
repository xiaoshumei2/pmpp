#include "../utils/common.cuh"

void matrixMulCPU2(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float total = 0.0f;
            for (int h = 0; h < K; h++) {
                total += A[row * K + h] * B[h * N + col];
            }
            C[row * N + col] = total;
        }
    }
}

void matrixMulTransposedCPU(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float total = 0.0f;
            for (int h = 0; h < K; h++) {
                total += A[row * K + h] * B[col * K + h];
            }
            C[row * N + col] = total;
        }
    }
}

__global__
void matrixMulNaiveKernel(const float *A, const float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float total = 0.0f;
        for (int h = 0; h < K; h++) {
            total += A[row * K + h] * B[h * N + col];
        }
        C[row * N + col] = total;
    }
}

__global__
void matrixMulTransposedNaiveKernel (const float *A, const float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float total = 0.0f;
        for (int h = 0; h < K; h++) {
            total += A[row * K + h] * B[col * K + h];
        }
        C[row * N + col] = total;
    }
}

#define TILE_WIDTH 32

__global__
void matrixMulTiledKernel(const float *A, const float *B, float *C, int M, int K, int N) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tileNums = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    float total = 0.0f;
    for (int tileIdx = 0; tileIdx < tileNums; tileIdx++) {
        int aCol = tileIdx * TILE_WIDTH + tx;
        int bRow = tileIdx * TILE_WIDTH + ty;
        tileA[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        for (int h = 0; h < TILE_WIDTH; h++) {
            total += tileA[ty][h] * tileB[h][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = total;
    }
}

__global__
void matrixMulTransposedTiledKernel(const float *A, const float *B, float *C, int M, int K, int N) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    
    int tileNums = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    float total = 0.0f;

    for (int tileIdx = 0; tileIdx < tileNums; tileIdx++) {
        int abCol = tileIdx * TILE_WIDTH + tx;
        int bRow = blockIdx.x * blockDim.x + ty;
        tileA[ty][tx] = (row < M && abCol < K) ? A[row * K + abCol] : 0.0f;
        tileB[ty][tx] = (bRow < N && abCol < K) ? B[bRow * K + abCol] : 0.0f;
        __syncthreads();

        for (int h = 0; h < TILE_WIDTH; h++) {
            total += tileA[ty][h] * tileB[tx][h];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = total;
    }
}

__global__ 
void matrixMulTransposedTiledUncornerKernel(const float *A, const float *B, float *C, int M, int K, int N) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int tileNums = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    float total = 0.0f;

    for (int tileIdx = 0; tileIdx < tileNums; tileIdx++) {
        int aCol = tileIdx * TILE_WIDTH + tx;
        int bRow = blockIdx.x * blockDim.x + tx;
        int bCol = tileIdx * TILE_WIDTH + ty;
        tileA[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[ty][tx] = (bRow < N && bCol < K) ? B[bRow * K + bCol] : 0.0f;
        __syncthreads();
        for (int h = 0; h < TILE_WIDTH; h++) {
            total += tileA[ty][h] * tileB[h][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = total;
    }
}

__global__
void matrixMulTransposedTiledAvoidBankConflictKernel(const float *A, const float *B, float *C, int M, int K, int N) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH+1];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    
    int tileNums = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    float total = 0.0f;

    for (int tileIdx = 0; tileIdx < tileNums; tileIdx++) {
        int abCol = tileIdx * TILE_WIDTH + tx;
        int bRow = blockIdx.x * blockDim.x + ty;
        tileA[ty][tx] = (row < M && abCol < K) ? A[row * K + abCol] : 0.0f;
        tileB[ty][tx] = (bRow < N && abCol < K) ? B[bRow * K + abCol] : 0.0f;
        __syncthreads();

        for (int h = 0; h < TILE_WIDTH; h++) {
            total += tileA[ty][h] * tileB[tx][h];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = total;
    }
}

void runExperiment(int M, int K, int N) {
    printf("\n----- M=%d, K=%d, N=%d ------\n", M, K, N);
    float naiveCpuMs = 0.0f, transposedCpuMs = 0.0f, naiveKernelMs = 0.0f, tiledKernelMs = 0.0f, 
        transposedKernelMs = 0.0f, transposedTiledKernelMs = 0.0f, transposedTileUncorneredKernelMs = 0.0f,
        transposedTiledAvoidBankConflictKernelMs = 0.0f;

    bool naiveKernelOk, tiledKernelOk, transposedKernelOk, transposedTiledKernelOk, 
        transposedTileUncorneredKernelOk, transposedTiledAvoidBankConflictKernelOk;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);
    float *A_h, *B_h, *C_h, *C_transposed, *C_compare;
    A_h = (float*)malloc(bytesA);
    B_h = (float*)malloc(bytesB);
    C_h = (float*)malloc(bytesC);
    C_compare = (float*)malloc(bytesC);
    C_transposed = (float*)malloc(bytesC);

    fillRandomFloat(A_h, (size_t)M*K);
    fillRandomFloat(B_h, (size_t)K*N);

    CpuTimer cpuTimer;
    cpuTimer.start();
    matrixMulCPU2(A_h, B_h, C_h, M, K, N);
    naiveCpuMs = cpuTimer.elapsedMs();

    cpuTimer.start();
    matrixMulTransposedCPU(A_h, B_h, C_transposed, M, K, N);
    transposedCpuMs = cpuTimer.elapsedMs();

    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, bytesA);
    cudaMalloc(&B_d, bytesB);
    cudaMalloc(&C_d, bytesC);
    cudaMemcpy(A_d, A_h, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytesB, cudaMemcpyHostToDevice);
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    GpuTimer gT1;
    gT1.start();
    matrixMulNaiveKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);
    gT1.stop();
    naiveKernelMs = gT1.elapsedMs();
    cudaMemcpy(C_compare, C_d, bytesC, cudaMemcpyDeviceToHost);
    naiveKernelOk = compareArrays(C_h, C_compare, (size_t)M*N, 1e-2f, false);

    GpuTimer gT2;
    gT2.start();
    matrixMulTiledKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);
    gT2.stop();
    tiledKernelMs = gT2.elapsedMs();
    cudaMemcpy(C_compare, C_d, bytesC, cudaMemcpyDeviceToHost);
    tiledKernelOk = compareArrays(C_h, C_compare, (size_t)M*N, 1e-2f, false);

    GpuTimer gT3;
    gT3.start();
    matrixMulTransposedNaiveKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);
    gT3.stop();
    transposedKernelMs = gT3.elapsedMs();
    cudaMemcpy(C_compare, C_d, bytesC, cudaMemcpyDeviceToHost);
    transposedKernelOk = compareArrays(C_transposed, C_compare, (size_t)M*N, 1e-2f, false);

    GpuTimer gT4;
    gT4.start();
    matrixMulTransposedTiledKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);
    gT4.stop();
    transposedTiledKernelMs = gT4.elapsedMs();
    cudaMemcpy(C_compare, C_d, bytesC, cudaMemcpyDeviceToHost);
    transposedTiledKernelOk = compareArrays(C_transposed, C_compare, (size_t)M*N, 1e-2f, false);

    GpuTimer gT5;
    gT5.start();
    matrixMulTransposedTiledUncornerKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);
    gT5.stop();
    transposedTileUncorneredKernelMs = gT5.elapsedMs();
    cudaMemcpy(C_compare, C_d, bytesC, cudaMemcpyDeviceToHost);
    transposedTileUncorneredKernelOk = compareArrays(C_transposed, C_compare, (size_t)M*N, 1e-2f, false);

    GpuTimer gT6;
    gT6.start();
    matrixMulTransposedTiledAvoidBankConflictKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);
    gT6.stop();
    transposedTiledAvoidBankConflictKernelMs = gT6.elapsedMs();
    cudaMemcpy(C_compare, C_d, bytesC, cudaMemcpyDeviceToHost);
    transposedTiledAvoidBankConflictKernelOk = compareArrays(C_transposed, C_compare, (size_t)M*N, 1e-2f, false);

    printf("naiveKernelOk = %d\n", naiveKernelOk);
    printf("tiledKernelOk = %d\n", tiledKernelOk);
    printf("transposedKernelOk = %d\n", transposedKernelOk);
    printf("transposedTiledKernelOk = %d\n", transposedTiledKernelOk);
    printf("transposedTileUncorneredKernelOk = %d\n", transposedTileUncorneredKernelOk);
    printf("transposedTiledAvoidBankConflictKernelOk = %d\n", transposedTiledAvoidBankConflictKernelOk);

    printf("naiveCpuMs = %10.3f ms\n", naiveCpuMs);
    printf("transposedCpuMs = %10.3f ms\n", transposedCpuMs);
    printf("naiveKernelMs = %10.3f ms\n", naiveKernelMs);
    printf("tiledKernelMs = %10.3f ms\n", tiledKernelMs);
    printf("transposedKernelMs = %10.3f ms\n", transposedKernelMs);
    printf("transposedTiledKernelMs = %10.3f ms\n", transposedTiledKernelMs);
    printf("transposedTileUncorneredKernelMs = %10.3f ms\n", transposedTileUncorneredKernelMs);
    printf("transposedTiledAvoidBankConflictKernelMs = %10.3f ms\n", transposedTiledAvoidBankConflictKernelMs);


    free(A_h); free(B_h); free(C_h); free(C_transposed); free(C_compare);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

int main() {
    runExperiment(1024, 1025, 1024);
    return 0;
}
