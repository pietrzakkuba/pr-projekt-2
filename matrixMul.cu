﻿// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

template <int BLOCK_SIZE> __global__ void MatrixMulCUDASample(float *A,
                                                        float *B, float *C, int WIDTH) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = WIDTH * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + WIDTH - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * WIDTH;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + WIDTH * ty + tx];
        Bs[ty][tx] = B[b + WIDTH * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = WIDTH * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + WIDTH * ty + tx] = Csub;
}

//template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float* C, float* A,
//	float* B, int wA,
//	int wB) {
//	// Block index
//	int bx = blockIdx.x;
//	int by = blockIdx.y;
//
//	// Thread index
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
//
//	int row = by * blockDim.y + ty;
//	int col = bx * blockDim.x + tx;
//
//	// Csub is used to store the element of the block sub-matrix
//	// that is computed by the thread
//	float Csub = 0;
//
//	// Multiply the two matrices together;
//	// each thread computes one element
//	// of the block sub-matrix
//#pragma unroll
//	for (int k = 0; k < wA; ++k) {
//		Csub += A[row * wA + k] * B[k * wB + col];
//	}
//
//	// Write the block sub-matrix to device memory;
//	// each thread writes one element
//	C[row * wB + col] = Csub;
//}



template <int BLOCK_SIZE> __global__ void MatrixMulKernel_2(float* A, float* B, float* C, int WIDTH)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // wyznaczenie indeksu wiersza/kolumny obliczanego elementu tablicy Cd
    int Row = bx * blockDim.y + ty;
    int Col = by * blockDim.x + tx;
    float C_local = 0;
    // każdy wątek z bloku oblicza jeden element macierzy
    for (int k = 0; k < WIDTH; ++k) C_local += A[Row * WIDTH + k] * B[k * WIDTH + Col];
    // zapis wyniku
    C[Row * WIDTH + Col] = C_local;
}

template <int BLOCK_SIZE> __global__ void MatrixMulKernel_3(float* Ad, float* Bd, float* Cd, int WIDTH)
{
    __shared__ float Ads[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bds[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Row = by * BLOCK_SIZE + ty;
    int Col = bx * BLOCK_SIZE + tx;
    float C_local = 0;
    // określenie obliczanego przez wątek elementu macierzy (jak w poprzednim kodzie – tu brak)
    //tx, ty to identyfikatory wątków w ramach bloku, Row i Col - analogicznie
    for (int m = 0; m < WIDTH / BLOCK_SIZE; ++m) {
        Ads[ty][tx] = Ad[Row * WIDTH + m * BLOCK_SIZE + tx]; //kolejny element dla sąsiedniego wątku
        Bds[ty][tx] = Bd[(m * BLOCK_SIZE + ty) * WIDTH + Col]; // używana kolumna – jakość pobrań ?
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
            C_local += Ads[ty][k] * Bds[k][tx];
        __syncthreads();
    }
    Cd[Row * WIDTH + Col] = C_local;
}

template <int BLOCK_SIZE> __global__ void MatrixMulKernel_4(float* Ad, float* Bd, float* Cd, int WIDTH)
{
    __shared__ float AAds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float ABds[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float BAds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float BBds[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Row = by * BLOCK_SIZE + ty;
    int Col = bx * BLOCK_SIZE + tx;

            
    int m = 0;
    AAds[ty][tx] = Ad[Row * WIDTH + m * BLOCK_SIZE + tx]; //kolejny element dla sąsiedniego wątku
    ABds[ty][tx] = Bd[(m * BLOCK_SIZE + ty) * WIDTH + Col]; // używana kolumna – jakość pobrań ?


    float C_local = 0;

    // określenie obliczanego przez wątek elementu macierzy (jak w poprzednim kodzie – tu brak)
    //tx, ty to identyfikatory wątków w ramach bloku, Row i Col - analogicznie
    for (m = 1; m < WIDTH / BLOCK_SIZE; ++m) {
        BAds[ty][tx] = AAds[ty][tx]; //kolejny element dla sąsiedniego wątku
        BBds[ty][tx] = ABds[ty][tx]; // używana kolumna – jakość pobrań ?
        __syncthreads();

        AAds[ty][tx] = Ad[Row * WIDTH + m * BLOCK_SIZE + tx];
        ABds[ty][tx] = Bd[(m * BLOCK_SIZE + ty) * WIDTH + Col];

        for (int k = 0; k < BLOCK_SIZE; ++k)
            C_local += BAds[ty][k] * BBds[k][tx];

        __syncthreads();
    }

    for (int k = 0; k < BLOCK_SIZE; ++k)
        C_local += AAds[ty][k] * ABds[k][tx];
    __syncthreads();

    Cd[Row * WIDTH + Col] = C_local;

}



void ConstantInit(float* data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char** argv,
    int block_size, const dim3& dimsA,
    const dim3& dimsB) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = reinterpret_cast<float*>(malloc(mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = reinterpret_cast<float*>(malloc(mem_size_B));
    cudaStream_t stream;

    // Initialize host memory
    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    // Allocate device memory
    float* d_A, * d_B, * d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float* h_C = reinterpret_cast<float*>(malloc(mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C));
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16) {
        MatrixMulKernel_4<16> << < grid, threads, 0, stream >> > (d_A, d_B, d_C,
            dimsA.x);
    }
    else {
        MatrixMulKernel_4<32> << < grid, threads, 0, stream >> > (d_A, d_B, d_C,
            dimsA.x);
    }

    printf("done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel
    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulKernel_4<16> << <grid, threads, 0, stream >> > (d_A, d_B, d_C,
                dimsA.x);
        }
        else {
            MatrixMulKernel_4<32> << <grid, threads, 0, stream >> > (d_A, d_B, d_C,
                dimsA.x);
        }
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
        static_cast<double>(dimsA.y) *
        static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
        (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops," \
        " WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6;  // machine zero

    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    printf("\nNOTE: The CUDA Samples are not meant for performance"\
        "measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    }
    else {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char** argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
        checkCmdLineFlag(argc, (const char**)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices" \
            " must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char**)argv);

    int block_size = 32;


    //dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    //dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsA(3072, 3072, 1);
    dim3 dimsB(3072, 3072, 1);

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
            dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
        dimsB.x, dimsB.y);

    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);

    exit(matrix_result);
}
