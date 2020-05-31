#define MATRIXSIZE 1024

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

template <int BLOCK_SIZE> __global__ void MatrixMulregisterCompute(float* Ad, float* Bd, float* Cd, int WIDTH)
{
    float AAds;
    float ABds;

    __shared__ float BAds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float BBds[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Row = by * BLOCK_SIZE + ty;
    int Col = bx * BLOCK_SIZE + tx;


    int m = 0;
    AAds = Ad[Row * WIDTH + m * BLOCK_SIZE + tx]; //kolejny element dla sąsiedniego wątku
    ABds = Bd[(m * BLOCK_SIZE + ty) * WIDTH + Col]; // używana kolumna – jakość pobrań ?


    float C_local = 0;

    // określenie obliczanego przez wątek elementu macierzy (jak w poprzednim kodzie – tu brak)
    //tx, ty to identyfikatory wątków w ramach bloku, Row i Col - analogicznie
    for (m = 1; m < WIDTH / BLOCK_SIZE; ++m) {
        BAds[ty][tx] = AAds; //kolejny element dla sąsiedniego wątku
        BBds[ty][tx] = ABds; // używana kolumna – jakość pobrań ?
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            C_local += BAds[ty][k] * BBds[k][tx];

        AAds = Ad[Row * WIDTH + m * BLOCK_SIZE + tx];
        ABds = Bd[(m * BLOCK_SIZE + ty) * WIDTH + Col];
        __syncthreads();
    }
    
    BAds[ty][tx] = AAds; //kolejny element dla sąsiedniego wątku
    BBds[ty][tx] = ABds;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k)
        C_local += BAds[ty][k] * BBds[k][tx];

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

    // Performs warmup operation using matrixMul CUDA kernel////////////
    // !!! ------------------------------------------------ !!!
    switch(block_size){
    case 8:     
        MatrixMulregisterCompute<8> << < grid, threads, 0, stream >> > (d_A, d_B, d_C, dimsA.x);
        break;
    case 16:    
        MatrixMulregisterCompute<16> << < grid, threads, 0, stream >> > (d_A, d_B, d_C, dimsA.x);
        break;
    case 32: 
        MatrixMulregisterCompute<32> << < grid, threads, 0, stream >> > (d_A, d_B, d_C, dimsA.x);
        break;
	}

    printf("done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel
    int nIter = 300;

    // !!! ------------------------------------------------ !!!

    switch(block_size){
    case 8:     
        for (int j = 0; j < nIter; j++) {
            MatrixMulregisterCompute<8> << <grid, threads, 0, stream >> > (d_A, d_B, d_C, dimsA.x);
        }
        break;
    case 16:    
        for (int j = 0; j < nIter; j++) {
            MatrixMulregisterCompute<16> << <grid, threads, 0, stream >> > (d_A, d_B, d_C, dimsA.x);
        }
        break;
    case 32: 
        for (int j = 0; j < nIter; j++) {
            MatrixMulregisterCompute<32> << <grid, threads, 0, stream >> > (d_A, d_B, d_C, dimsA.x);
        }
        break;
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

    int block_size = std::stoi(argv[1]);
    printf("Rozmiar bloku: %d\n", block_size);
    printf("Algorytm: registerCompute\n");

    dim3 dimsA(MATRIXSIZE, MATRIXSIZE, 1);
    dim3 dimsB(MATRIXSIZE, MATRIXSIZE, 1);

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