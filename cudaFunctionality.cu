#include "cudaFunctionality.hpp"
#include <iostream>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <bitset>
#include "CudaTimer.cuh"

// if hamming distance is 0 then return 0, if is 1 return 1, else return 2
__device__ __host__ int _hammingDistance(unsigned int a, unsigned int b)
{
    int x = a ^ b;
    if ( 0 == x )
        return 0;
    if ( 0 == (x & (x - 1)) ) // check if x is power of 2
        return 1; 
    return 2;
}

__device__ __host__ bool _HammingDistanceOne(unsigned int *v1, unsigned int *v2, int VLength)
{
    int hamdis = 0;
    for (int i = 0; i < VLength; i++)
    {
        hamdis += _hammingDistance(v1[i], v2[i]);
        if (hamdis > 1)
            return false;
    }

    return hamdis == 1;
}

__device__ bool _HammingDistanceOne2(unsigned int *v1, unsigned int *v2, int VLength)
{
    int hamdis = 0;
    for (int i = 0; i < VLength; i++)
    {
        hamdis += __popc(v1[i] ^ v2[i]);
        if (hamdis > 1)
            return false;
    }

    return hamdis == 1;
}

void InitializeResultCUDA(unsigned int **&out, int N, int VLength)
{
    unsigned int **tempOut = new  unsigned int*[N]; 

    for(int i = 0; i < N; i++){
        cudaMalloc(&tempOut[i], VLength * sizeof(unsigned int));
        cudaMemset(tempOut[i], 0, VLength * sizeof(unsigned int));
    }

    cudaMalloc(&out, N * sizeof(unsigned int *));
    cudaMemcpy(out, tempOut, N * sizeof(unsigned int*), ::cudaMemcpyHostToDevice);
}

void freeCudaResults(unsigned int **&d_res, int N)
{
    if (nullptr != d_res)
        return;

    for(int i = 0; i < N; i++){
        if(nullptr != d_res[i])
            cudaFree(d_res[i]);
    }

    cudaFree(d_res);
    d_res = nullptr;
}

void CopyCudaResultToHost(unsigned int **d_out, unsigned int **h_out, int N, int VLength)
{
    unsigned int ** tempArr = new unsigned int*[N];
    cudaMemcpy(tempArr, d_out, N * sizeof(unsigned int*), ::cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; i++){
        cudaMemcpy(h_out[i], tempArr[i], VLength * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
    }

    delete [] tempArr;
}

__global__ void naiveKernel3(unsigned int **data, int N, int VLength, unsigned int *out, int out_stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N)
        return;

    int stride_tid = out_stride * tid;
    unsigned int *vector = data[tid];

    for (int i = N - 1; i > tid; i--)
    {
        if (_HammingDistanceOne2(vector, data[i], VLength))
        {
            int offset_i = i / VECTOR_BLOCK_BITS;
            out[stride_tid + offset_i] |= 1 << (VECTOR_BLOCK_BITS - 1 - i % VECTOR_BLOCK_BITS);
        }
    }
}

bool HammingDistanceOne(unsigned int *v1, unsigned int *v2, int VLength)
{
    return _HammingDistanceOne(v1, v2, VLength);
}

void LaunchNaiveCuda3(unsigned int **d_vectors, int N, int VLength, unsigned int *d_out)
{
    int threads = min(N, MAX_THREAD_COUNT);
    int blocks = intDivCeil(N, threads);
    int out_stride = intDivCeil(N, VECTOR_BLOCK_BITS);

    if (blocks * threads < N)
        blocks++;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    naiveKernel3<<<blocks, threads>>>(d_vectors, N, VLength, d_out, out_stride);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    float time = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    time += milliseconds;
    std::cout << std::fixed << "It took " << time << "ms" << std::endl;
}

void copyVectorSetToCuda(VectorSet *vectorSet, unsigned int **d_data)
{
    unsigned int **tempArr = new unsigned int *[vectorSet->N];

    cudaMemcpy(tempArr, d_data, vectorSet->N * sizeof(unsigned int *), ::cudaMemcpyDeviceToHost);

    for (int i = 0; i < vectorSet->N; i++)
    {
        cudaMemcpyAsync(tempArr[i], vectorSet->data[i], vectorSet->VLength * sizeof(int), ::cudaMemcpyHostToDevice);
    }

    cudaStreamSynchronize(0);

    delete[] tempArr;
}

void initializeCudaData(unsigned int **&d_data, VectorSet *vectorSet)
{
    cudaMalloc((void **)(&d_data), vectorSet->N * sizeof(unsigned int *));
    unsigned int **tempArr = new unsigned int *[vectorSet->N];

    for (int i = 0; i < vectorSet->N; i++)
    {
        cudaMalloc(&tempArr[i], vectorSet->VLength * sizeof(int));
    }

    cudaMemcpy(d_data, tempArr, vectorSet->N * sizeof(unsigned int *), ::cudaMemcpyHostToDevice);

    delete[] tempArr;

    copyVectorSetToCuda(vectorSet, d_data);
}
void freeCudaData(unsigned int **&d_data, int N)
{
    if (nullptr == d_data)
        return;

    unsigned int **tempArr = new unsigned int *[N];

    cudaMemcpy(tempArr, d_data, N * sizeof(int *), ::cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
    {
        cudaFree(tempArr[i]);
    }
    delete[] tempArr;

    cudaFree(d_data);

    d_data = nullptr;
}

// Works when x != 0
__device__ __host__ int _intDivCeil(int x, int y)
{
    return 1 + ((x - 1) / y);
}

int intDivCeil(int x, int y)
{
    return _intDivCeil(x, y);
}

__global__ void kernelTransposeData(unsigned int ** data, unsigned int ** keys, int Nx, int Ny){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= Nx || y >= Ny) return;

    keys[x][y] = data[y][x];
}

__global__ void kernelGetKeys(unsigned int ** data, unsigned int * keys, int N, int i){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= N) return;
    keys[x] = data[x][i];

}

void _PrintVectorInBinary(unsigned int * vec, int N){
    for(int i = 0; i < N; i++){
        std::cout << std::bitset<8 * sizeof(int)>(vec[i]);
    }
    std::cout << std::endl;
}

void getKeys(unsigned int ** data, unsigned int * keyArr, int N, int i){
    int threads = min(N, MAX_THREAD_COUNT);
    int blocks = intDivCeil(N, threads);

    kernelGetKeys<<<blocks, threads>>>(data, keyArr, N, i);
}

void sortVectors(unsigned int ** data, int N, int VL){
    unsigned int * keyArr;
    cudaMalloc(&keyArr, N * sizeof(unsigned int));
    
    for(int i = VL - 1; i >= 0; i--){
        getKeys(data, keyArr, N, i);
        thrust::stable_sort_by_key(thrust::device, keyArr, keyArr + N, data);
    }

    cudaFree(keyArr);
}

__global__ void kernelGetRSTKeys(unsigned int ** data, unsigned int * keys, int N, int i, unsigned int mask, int shift){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= N) return;
    keys[x] = (data[x][i] & mask) >> shift;
}

__global__ void kernelSetChildFound(unsigned int * child_marked, unsigned int * P, unsigned int * keys, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= N) return;

    child_marked[P[tid] * 4 + keys[tid]] = 1;
}

__global__ void kernelUpdateP(unsigned int * prescan, unsigned int * P, unsigned int * keys, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= N) return;

    P[tid] = prescan[P[tid] * 4 + keys[tid]];    
}

__global__ void kernelUpdateLayer(unsigned int * prescan, RSTLayer * layer, unsigned int * child_marked, unsigned int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= N) return;

    layer->idx[tid] = child_marked[tid] == 1 ? (int)prescan[tid] : -1;
}

__device__ bool isVectorInTree(RSTLayer * tree, unsigned int * vector, int VL){
    int nodeID = 0;
    RSTLayer * treeLayer = tree;

    for(int i = 0; i < VL; i++){
        unsigned int mask = DEFAULT_MASK;
        int shift = DEFAULT_TREE_SHIFT;
        unsigned int key = vector[i];
        for(int j = 0; j < LAYERS_IN_BLOCK; j++){
            int treeKey = (key & mask) >> shift;
            int nextNodeID = treeLayer->idx[nodeID * 4 + treeKey];

            if(-1 == nextNodeID)
                return false;

            nodeID = nextNodeID;

            shift -= TREE_DEGREE_EXP;
            mask >>= TREE_DEGREE_EXP;
            treeLayer++;
        }
    }

    return true;

}

void setChildFound(unsigned int * child_marked, unsigned int * P, unsigned int * keys, int N){
    int threads = min(N, MAX_THREAD_COUNT);
    int blocks = intDivCeil(N, threads);

    kernelSetChildFound<<<blocks, threads>>>(child_marked, P, keys, N);
}


void getRSTKeys(unsigned int ** data, unsigned int * keyArr, int N, int i, unsigned int mask, int shift){
    int threads = min(N, MAX_THREAD_COUNT);
    int blocks = intDivCeil(N, threads);

    kernelGetRSTKeys<<<blocks, threads>>>(data, keyArr, N, i, mask, shift);
}


void freeRSTCuda(struct RSTLayer *& tree, int N, int VL){
    int num_of_layers = VL * LAYERS_IN_BLOCK + 1;
    RSTLayer * tempTree = new RSTLayer[num_of_layers];
    cudaMemcpy(tempTree, tree, num_of_layers * sizeof(RSTLayer), ::cudaMemcpyDeviceToHost);

    for(int i = 0; i < num_of_layers; i++){
        cudaFree(tempTree[i].idx);
    }
    cudaFree(tree);
    delete [] tempTree;
    tree = nullptr;
}

void buildRSTCuda(unsigned int ** data, int N, int VL, int BL, struct RSTLayer *& tree){    
    int threads = min(N, MAX_THREAD_COUNT);
    int blocks = intDivCeil(N, threads);
    int child_threads, child_blocks;

    int layers_in_block = LAYERS_IN_BLOCK;
    int num_of_layers = VL * LAYERS_IN_BLOCK + 1;

    cudaMalloc(&tree, num_of_layers * sizeof(RSTLayer));

    struct RSTLayer * treeRef = tree;

    unsigned int * keys, *P, *prescan;
    unsigned int * child_marked;
    cudaMalloc(&keys, N * sizeof(unsigned int));

    int children = 1;
    int children_refs = children * TREE_DEGREE;
    int child_marked_size = 10 * TREE_DEGREE;
    
    unsigned int isLastMarked = false;

    RSTLayer * tempLayer = new RSTLayer();

    cudaMalloc(&tempLayer->idx, sizeof(int) * children_refs);
    cudaMemcpy(tree, tempLayer, sizeof(RSTLayer), ::cudaMemcpyHostToDevice);

    cudaMalloc(&child_marked, child_marked_size * sizeof(unsigned int));
    cudaMalloc(&prescan, child_marked_size * sizeof(unsigned int));

    cudaMalloc(&P, N * sizeof(unsigned int));
    cudaMemset(P, 0, N * sizeof(unsigned int));
    int layerID = 0;
    for(int i = 0; i < VL; i++){
        unsigned int mask = DEFAULT_MASK;
        int shift = DEFAULT_TREE_SHIFT;
        for(int j = 0 ; j < layers_in_block; j++){
            // get keys for current layer
            kernelGetRSTKeys<<<blocks, threads>>>(data, keys, N, i, mask, shift);
            cudaDeviceSynchronize();

            // mark children as found according to blocks (P array)
            kernelSetChildFound<<<blocks, threads>>>(child_marked, P, keys, N);
            cudaDeviceSynchronize();

            // prescan of marked children array
            thrust::exclusive_scan(thrust::device, child_marked, child_marked + children_refs, prescan);
            cudaDeviceSynchronize();
            
            // update block assignment
            kernelUpdateP<<<blocks, threads>>>(prescan, P, keys, N);
            cudaDeviceSynchronize();


            child_threads = min(children_refs, MAX_THREAD_COUNT);
            child_blocks = intDivCeil(children_refs, child_threads);

            // update id refs to next layer
            kernelUpdateLayer<<<child_blocks, child_threads>>>(prescan, treeRef, child_marked, children_refs);

            cudaDeviceSynchronize();

            // calculate new children value
            cudaMemcpy(&isLastMarked, child_marked+(children_refs - 1), sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
            cudaMemcpy(&children, (prescan+(children_refs - 1)), sizeof(unsigned int), ::cudaMemcpyDeviceToHost);

            if(isLastMarked){
                children++;
            }
            children_refs = children * TREE_DEGREE;

            //create next layer
            cudaMalloc(&tempLayer->idx, sizeof(int) * children_refs);

            cudaMemcpy(treeRef + 1, tempLayer, sizeof(RSTLayer), ::cudaMemcpyHostToDevice);

            // increase arrays size if are insufficient
            if(children_refs >= child_marked_size){
                cudaFree(child_marked);
                cudaFree(prescan);

                child_marked_size = children_refs * 3;

                cudaMalloc(&child_marked, child_marked_size * sizeof(unsigned int));
                cudaMalloc(&prescan, child_marked_size * sizeof(unsigned int));

            }

            cudaMemset(child_marked, 0, sizeof(unsigned int) * children_refs);

            shift -= TREE_DEGREE_EXP;
            mask >>= TREE_DEGREE_EXP;
            treeRef++;
            layerID++;

        }
    }

    cudaFree(child_marked);
    cudaFree(prescan);
    cudaFree(P);
    delete tempLayer;
}

__global__ void rstKernel(unsigned int ** data, RSTLayer * tree, int N, int VLength, int BL, unsigned int* pairCount, unsigned int** out){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int foundPairs = 0;
    if (tid >= N)
        return;

    unsigned int *vector = data[tid];
    unsigned int *outVector = out[tid];
    unsigned int out_block;
    int idx = 0;
    for(int i = 0; i < VLength; i++){
        unsigned int mask = STARTING_BIT_MASK;
        out_block = 0;
        for(int j = 0; j < VECTOR_BLOCK_BITS; j++){
            if(idx >= BL){
                break;
            }

            vector[i] ^= mask;

            if(isVectorInTree(tree, vector, VLength)){
                foundPairs++;
                out_block |= mask;
            }

            vector[i] ^= mask;
            
            mask >>= 1;
            idx++;
        }

        outVector[i] = out_block;
    }
    
    if(foundPairs)
        atomicAdd(pairCount, foundPairs);
}

void LaunchRSTCuda(unsigned int **d_vectors, int N, int VLength, int BLength, RSTLayer * tree, unsigned int &foundPairs, unsigned int ** out){
    int threads = min(N, MAX_THREAD_COUNT);
    int blocks = intDivCeil(N, threads);

    unsigned int *d_foundPairs;
    cudaMalloc(&d_foundPairs, sizeof(unsigned int));
    cudaMemset(d_foundPairs, 0, sizeof(unsigned int));
    rstKernel<<<blocks, threads>>>(d_vectors, tree, N, VLength, BLength, d_foundPairs, out);
    cudaMemcpy(&foundPairs, d_foundPairs, sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
    cudaFree(d_foundPairs);
}

void HammingOneCuda(unsigned int **&d_data, unsigned int **&d_out, VectorSet * vectorSet, unsigned int &gpuFoundPairs){
    CudaTimer cuTimer;
    RSTLayer * tree;

    std::cout << "INITIALIZE CUDA RESULT STARTED" << std::endl;
    cuTimer.Start();
    InitializeResultCUDA(d_out, vectorSet->N, vectorSet->VLength);
    cuTimer.Stop();
    std::cout << "INITIALIZE CUDA RESULT FINISHED. It took: " << cuTimer.GetElapsedTime() << "ms" << std::endl;

    std::cout << "INITIALIZE CUDA DATA STARTED" << std::endl;
    cuTimer.Start();
    initializeCudaData(d_data, vectorSet);
    cuTimer.Stop();
    std::cout << "INITIALIZE CUDA DATA FINISHED. It took: " << cuTimer.GetElapsedTime() << "ms" << std::endl;

    std::cout << "CUDA DATA SORT STARTED" << std::endl;
    cuTimer.Start();
    sortVectors(d_data, vectorSet->N, vectorSet->VLength);
    cuTimer.Stop();
    std::cout << "CUDA DATA SORT FINISHED. It took: " << cuTimer.GetElapsedTime() << "ms" << std::endl;
    
    std::cout << "CUDA RST BUILD STARTED" << std::endl;
    cuTimer.Start();
    buildRSTCuda(d_data, vectorSet->N, vectorSet->VLength, vectorSet->BLength, tree);
    cuTimer.Stop();
    std::cout << "CUDA RST BUILD FINISHED. It took: " << cuTimer.GetElapsedTime() << "ms" << std::endl;
    
    std::cout << "CUDA HAMMING_ONE-RST STARTED" << std::endl;
    cuTimer.Start();
    LaunchRSTCuda(d_data, vectorSet->N, vectorSet->VLength, vectorSet->BLength, tree, gpuFoundPairs, d_out);
    cuTimer.Stop();
    std::cout << "CUDA HAMMING_ONE-RST FINISHED. It took: " << cuTimer.GetElapsedTime() << "ms" << std::endl;
    std::cout << "Found " << gpuFoundPairs << " pairs with CUDA" << std::endl;

    freeRSTCuda(tree, vectorSet->N, vectorSet->VLength);

}