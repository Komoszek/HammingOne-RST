#pragma once
#ifndef _DATALOADER_HPP_
#define _DATALOADER_HPP_

#include <iostream>
#include <string>
#include "VectorSet.hpp"
#include <fstream>
#include "Defines.hpp"
#include <random>
#include <bitset>
#include <string>

void PrintVectorInBinary(unsigned int * vec, int N){
    for(int i = 0; i < N; i++){
        std::cout << std::bitset<8 * sizeof(int)>(vec[i]);
    }
    std::cout << std::endl;
}

void PrintSet(VectorSet * set) {
    for(int i = 0; i < set->N; i++){
        PrintVectorInBinary(set->data[i], set->VLength);
    }
}

bool LoadSequences(std::string path, VectorSet *&set)
{
    std::ifstream file(path, std::fstream::binary);

    if (!file.is_open())
    {
        std::cout << "Unable to open file " << path << std::endl;
        file.close();
        return false;
    }

    int headerData[2];

    file.read((char *)headerData, sizeof(int) * 2);

    std::cout << headerData[0] << " " << headerData[1] << std::endl;

    if (nullptr != set)
        delete set;

    set = new VectorSet(headerData[0], headerData[1]);

    for (int i = 0; i < set->N; i++)
    {
        file.read((char *)set->data[i], VECTOR_BLOCK_BYTES * set->VLength);
        if (file.fail() || file.eof())
        {
            delete set;
            file.close();
            return false;
        }
    }

    file.close();
    return true;
}

bool SaveSequences(std::string path, VectorSet *set)
{
    std::ofstream file(path, std::fstream::binary);

    if (!file.is_open())
    {
        std::cout << "Unable to open file" << path << std::endl;
        file.close();
        return false;
    }

    int headerData[2];
    headerData[0] = set->N;
    headerData[1] = set->BLength;

    file.write((char *)headerData, sizeof(int) * 2);

    for (int i = 0; i < set->N; i++)
    {
        file.write((char *)set->data[i], VECTOR_BLOCK_BYTES * set->VLength);
    }

    file.close();
    return true;
}

void GenerateRandomData(int n, int l, VectorSet *&set, int seed)
{
    if (nullptr != set)
        delete set;

    set = new VectorSet(n, l);

    int remainder = (set->VLength * VECTOR_BLOCK_BITS - set->BLength);

    unsigned int remainderMask = ~((1 << remainder) - 1);

    std::mt19937 gen(seed);
    std::uniform_int_distribution<unsigned int> distByte(0, (unsigned int)~0);

    for (int k = 0; k < set->N; k++)
    {
        for (int i = 0; i < set->VLength; i++)
        {
            set->data[k][i] = distByte(gen);
        }
        
        set->data[k][set->VLength - 1] &= remainderMask;
    }
}

bool saveResults(unsigned int **data, unsigned int **out, int N, int VLength, int BLength, std::string path){
    std::ofstream file(path);

    if (!file.is_open())
    {
        std::cout << "Unable to open file" << path << std::endl;
        file.close();
        return false;
    }

    std::cout << "Saving results..." << std::endl;
    for(int i = 0; i < N; i++){
        int idx = 0;
        unsigned int * vector = data[i];
        unsigned int * outVector = out[i];

        for(int j = 0; j < VLength; j++){
            unsigned int mask = STARTING_BIT_MASK;
            for(int k = 0; k < VECTOR_BLOCK_BITS; k++){
                if(idx >= BLength) break;

                if((outVector[j] & mask) == mask){
                    for(int l = 0; l < VLength; l++){
                        file << std::bitset<VECTOR_BLOCK_BITS>(vector[l]);
                    }
                    file << " ; ";

                    vector[j] ^= mask;
                    for(int l = 0; l < VLength; l++){
                        file << std::bitset<VECTOR_BLOCK_BITS>(vector[l]);
                    }
                    vector[j] ^= mask;
                    file << std::endl;
                }
                mask >>= 1;
                idx++;
            }
        }
    }

    file.close();
    std::cout << "Results saved to " << path << std::endl;

    return true;
}


#endif