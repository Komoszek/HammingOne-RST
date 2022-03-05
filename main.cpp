#include <iostream>
#include <unistd.h>
#include "Config.hpp"
#include "cudaFunctionality.hpp"

#include "DataLoader.hpp"
#include "VectorSet.hpp"
#include "HammingCPU.hpp"
#include "CudaTimer.cuh"
#include <cstring>
#include <chrono>

bool AreResultsEqual(unsigned int * out1, unsigned int * out2, int N){
    return memcmp(out1, out2, N) == 0;
}

void usage(char * name){
    std::cout << "USAGE: " << name << "" << std::endl;
    std::cout << "-i: input file path for binary vectors" << std::endl;
    std::cout << "-o: output file path for generated data" << std::endl;
    std::cout << "-r: result file path for found pairs" << std::endl;
    std::cout << "-s: skip CPU calculations" << std::endl;
    std::cout << "-n: number of vectors (generate only)" << std::endl;
    std::cout << "-l: length of vectors (in bits; generate only)" << std::endl;
    std::cout << "-v: print set" << std::endl;

    exit(EXIT_FAILURE);
}

void parseArgs(int argc, char **argv, Config &config){
    char c;

    while ((c = getopt (argc, argv, "hi:o:r:sn:l:v")) != -1)
		switch (c){
            case 'i':
                config.seqInputPath = optarg;
                break;
            case 'o':
                config.seqOutputPath = optarg;
                break;
            case 'r':
                config.resultPath = optarg;
                break;
            case 'n':
                config.n = atoi(optarg);
                break;
            case 'l':
                config.l = atoi(optarg);
                break;
            case 's':
                config.skipCPU = true;
                break;
            case 'v':
                config.printSet = true;
                break;
			case '?':
            case 'h':
			default: usage(argv[0]);
		}
	if(argc>optind)usage(argv[0]);
}

int main(int argc, char **argv)
{
    CudaTimer cuTimer; 
    
    Config config;
    VectorSet * vectorSet = nullptr;
    parseArgs(argc, argv, config);

    if(!config.seqInputPath.empty()){
        std::cout << "DATA LOAD START" << std::endl;
        cuTimer.Start();
        LoadSequences(config.seqInputPath, vectorSet);
        cuTimer.Stop();
        std::cout << "DATA LOAD FINISHED. It took: " << cuTimer.GetElapsedTime() << "ms" << std::endl;
    } else {
        std::cout << "DATA GENERATE START" << std::endl;
        cuTimer.Start();
        GenerateRandomData(config.n, config.l, vectorSet, RANDOM_SEED);
        cuTimer.Stop();
        std::cout << "DATA GENERATE FINISHED. It took: " << cuTimer.GetElapsedTime() << "ms" << std::endl;
    }

    if(!config.seqOutputPath.empty()){
        SaveSequences(config.seqOutputPath, vectorSet);
    }
    
    unsigned int ** out = nullptr, ** d_out = nullptr, ** d_data = nullptr;
    unsigned int gpuFoundPairs = 0;

    if(config.printSet)
        PrintSet(vectorSet);

    HammingOneCuda(d_data, d_out, vectorSet, gpuFoundPairs);
    
    if(!config.skipCPU){
        unsigned int pairsCount;
        pairsCount = HammingOneCPU(out, vectorSet);
        std::cout << "Found " << pairsCount << " pairs with CPU" << std::endl;
        if(pairsCount == gpuFoundPairs){
            std::cout << "OK, Results match" << std::endl;
        } else {
            std::cout << "ERR, Results don't match" << std::endl;
        }
    }

    if(!config.resultPath.empty()){
        unsigned int ** h_out, **h_data;
        InitializeResultCpu(h_out, vectorSet->N, vectorSet->VLength);
        InitializeResultCpu(h_data, vectorSet->N, vectorSet->VLength);
        CopyCudaResultToHost(d_out, h_out, vectorSet->N, vectorSet->VLength);
        CopyCudaResultToHost(d_data, h_data, vectorSet->N, vectorSet->VLength);

        saveResults(h_data, h_out, vectorSet->N, vectorSet->VLength, vectorSet->BLength, config.resultPath);

        clearResultCpu(h_out, vectorSet->N);
        clearResultCpu(h_data, vectorSet->N);
    }
    
    
    freeCudaData(d_data, vectorSet->N);
    freeCudaResults(d_out, vectorSet->N);

    if(nullptr != out){
       clearResultCpu(out, vectorSet->N);
    }

    if(nullptr != vectorSet)
        delete vectorSet;
    

    return 0;
}
