#ifndef _CUDA_FUNCTIONALITY_HPP_
#define _CUDA_FUNCTIONALITY_HPP_

#include <cmath>
#include "Defines.hpp"
#include "VectorSet.hpp"


struct RSTLayer {
    int * idx;
};

int GetOutputLength(int N);
void InitializeResultCUDA(unsigned int **&out, int N, int VLength);
void CopyCudaResultToHost(unsigned int ** d_out, unsigned int ** h_out, int N, int VLength);

bool HammingDistanceOne(unsigned int * v1, unsigned int * v2, int VLength);
void initializeCudaData(unsigned int **&d_data, VectorSet * VectorSet);
void copyVectorSetToCuda(VectorSet * vectorSet, unsigned int ** d_data);
void freeCudaData(unsigned int **& d_data, int N);
void freeCudaResults(unsigned int **&d_res, int N);

int intDivCeil(int x, int y);
int GetOutputLengthBlocks(int N);
void sortVectors(unsigned int ** data, int N, int VL);
void buildRSTCuda(unsigned int ** data, int N, int VL, int BL, struct RSTLayer *& tree);
void freeRSTCuda(struct RSTLayer *& tree, int N, int VL);
void LaunchRSTCuda(unsigned int **d_vectors, int N, int VLength, int BLength, RSTLayer * tree, unsigned int &foundPairs, unsigned int ** out);
void HammingOneCuda(unsigned int **&d_data, unsigned int **&d_out, VectorSet * vectorSet, unsigned int &gpuFoundPairs);

#endif