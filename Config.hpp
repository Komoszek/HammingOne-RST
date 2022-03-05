#ifndef _CONFIG_HPP_
#define _CONFIG_HPP
#include <string>

struct Config {
    std::string seqInputPath;
    std::string seqOutputPath;
    std::string resultPath;
    int n;
    int l;
    bool printSet;
    bool skipCPU;

    Config(){
        seqInputPath = "";
        seqOutputPath = "";
        resultPath = "";
        n = 10000;
        l = 1000;
        skipCPU = false;
        printSet = false;

    }
};

#endif