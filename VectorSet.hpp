#ifndef _VECTORSET_HPP_
#define _VECTORSET_HPP_

#include "Defines.hpp"
class VectorSet {
    public:
        int N;
        int BLength; // Length of vector in bits
        int VLength; // Length of vector in whole blocks (i.e. unsigned ints)
        unsigned int ** data;
        VectorSet(int n, int l){
            N = n;
            BLength = l;
            VLength = BLength / VECTOR_BLOCK_BITS;
            if(BLength % VECTOR_BLOCK_BITS)
                VLength++;

            data = new unsigned int*[N];

            for(int i = 0; i < N; i++)
                data[i] = new unsigned int[VLength];
        } 

        ~VectorSet(){
            for(int i = 0; i < N; i++){
                delete [] data[i];
            }

            delete [] data;
        }
};

#endif