#ifndef _HAMMING_HPP_
#define _HAMMINH_HPP_

#include "VectorSet.hpp"
#include "cudaFunctionality.hpp"
#include "DataLoader.hpp"
#include "Defines.hpp"
#include <iostream>
#include "CudaTimer.cuh"
#include <cstring>

struct RSTNodeCPU {
    RSTNodeCPU ** nodes;

    RSTNodeCPU() {
        nodes = new RSTNodeCPU*[TREE_DEGREE];
        for(unsigned int i = 0; i < TREE_DEGREE; i++){
            nodes[i] = nullptr;
        }
    }

    ~RSTNodeCPU() {
        for(unsigned int i = 0; i < TREE_DEGREE; i++){
            if(nullptr != nodes[i])
                delete nodes[i];
        }

        delete [] nodes;
    }
};

bool isVectorInRSTCPU(RSTNodeCPU * root, unsigned int * vector, int VLength){
    RSTNodeCPU * treeIt = root;

    for(int i = 0; i < VLength; i++){
        unsigned int mask = DEFAULT_MASK;
        int shift = DEFAULT_TREE_SHIFT;
        for(int j = 0; j < LAYERS_IN_BLOCK; j++){
            unsigned int key = (vector[i] & mask) >> shift;
            if(nullptr == treeIt->nodes[key]){
                return false;
            }

            treeIt = treeIt->nodes[key];

            shift -= TREE_DEGREE_EXP;
            mask >>= TREE_DEGREE_EXP;
        }
    }

    return true;
}


void insertRSTVectorCPU(RSTNodeCPU *&root, unsigned int * vector, int VLength){
    RSTNodeCPU * treeIt = root;

    for(int i = 0; i < VLength; i++){
        unsigned int mask = DEFAULT_MASK;
        int shift = DEFAULT_TREE_SHIFT;
        for(int j = 0; j < LAYERS_IN_BLOCK; j++){
            unsigned int key = (vector[i] & mask) >> shift;
            if(nullptr == treeIt->nodes[key]){
                treeIt->nodes[key] = new RSTNodeCPU();
            }

            treeIt = treeIt->nodes[key];

            shift -= TREE_DEGREE_EXP;
            mask >>= TREE_DEGREE_EXP;
        }
    }
}

void buildRSTCPU(VectorSet * vectorSet, RSTNodeCPU *&root){
    if(nullptr != root){
        delete root;
    } 
    
    root = new RSTNodeCPU();


    for(int i = 0; i < vectorSet->N; i++){
        insertRSTVectorCPU(root, vectorSet->data[i], vectorSet->VLength);
    }
}

int GetPairsRST(VectorSet * vectorSet, RSTNodeCPU * root, unsigned int ** out){
    int pairsCount = 0;
    for(int i = 0; i < vectorSet->N; i++){
        int idx = 0;
        unsigned int * vector = vectorSet->data[i];
        unsigned int * outVector = out[i];
        for(int j = 0; j < vectorSet->VLength; j++){
            unsigned int mask = STARTING_BIT_MASK;
            for(int k = 0; k < VECTOR_BLOCK_BITS; k++){
                if(idx >= vectorSet->BLength) break;

                vector[j] ^= mask;

                if(isVectorInRSTCPU(root, vector, vectorSet->VLength)){
                    outVector[j] &= mask;
                    pairsCount++;
                }

                vector[j] ^= mask;
                mask >>= 1;
                idx++;
            }
        }
    }

    return pairsCount;
}


void GetPairsMatrix(VectorSet * set, unsigned int * out){
    int stride = intDivCeil(set->N, VECTOR_BLOCK_BITS);
    int offset_i = 0;
    for(int i = 0; i < set->N; i++){
        for(int j = i + 1; j < set->N; j++){
            if(HammingDistanceOne(set->data[i], set->data[j], set->VLength) == 1){
                int offset2 = j / VECTOR_BLOCK_BITS;
                out[i * stride + offset2] |= 1 << (VECTOR_BLOCK_BITS - j % VECTOR_BLOCK_BITS - 1);
            } 
        }
        offset_i += stride;
    }
}


void InitializeResultCpu(unsigned int **& out, int N, int VLength){
    out = new unsigned int*[N];
    for(int i = 0; i < N; i++){
        out[i] = new unsigned int[VLength];
        memset(out[i], 0, VLength * sizeof(unsigned int));
    }
}


int HammingOneCPU(unsigned int **& out, VectorSet * vectorSet){
    CudaTimer cuTimer; 
    int pairsCount;

    std::cout << "INITIALIZE CPU RESULT START" << std::endl;
    cuTimer.Start();
    InitializeResultCpu(out, vectorSet->N, vectorSet->VLength);
    cuTimer.Stop();
    std::cout << "INITIALIZE CPU RESULT FINISHED. It took: " << cuTimer.GetElapsedTime() << "ms" << std::endl;

    RSTNodeCPU * tree = nullptr;
    std::cout << "BUILD CPU RST START" << std::endl;
    cuTimer.Start();

    buildRSTCPU(vectorSet, tree);
    cuTimer.Stop();
    std::cout << "BUILD CPU RST FINISHED. It took: " << cuTimer.GetElapsedTime() << "ms" << std::endl;

    std::cout << "CPU START" << std::endl;
    cuTimer.Start();
    pairsCount = GetPairsRST(vectorSet, tree, out);

    cuTimer.Stop();
    std::cout << "CPU FINISHED. It took: " << cuTimer.GetElapsedTime() << "ms" << std::endl;

    return pairsCount;
}

void clearResultCpu(unsigned int **& out, int N){
    if(nullptr == out) return;

    for(int i = 0; i < N; i++){
        if(nullptr != out[i]){
            delete [] out[i];
        }
    }

    delete [] out;
}




#endif