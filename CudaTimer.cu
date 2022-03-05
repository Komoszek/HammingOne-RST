#include "CudaTimer.cuh"

struct PrivateTimingCUDA{
    cudaEvent_t start; 
    cudaEvent_t stop;
};

CudaTimer::CudaTimer(){
    milliseconds = 0.0f;
    timings = new PrivateTimingCUDA;
    cudaEventCreate(&timings->start);
    cudaEventCreate(&timings->stop);
}

CudaTimer::~CudaTimer(){
    cudaEventDestroy(timings->start);
    cudaEventDestroy(timings->stop);
    delete timings;
}

void CudaTimer::Start(){
    milliseconds = 0.0f;
    cudaEventRecord(timings->start);
}

void CudaTimer::Stop(){
    cudaEventRecord(timings->stop);
    cudaEventSynchronize(timings->stop);
    cudaEventElapsedTime(&milliseconds, timings->start, timings->stop);
}

float CudaTimer::GetElapsedTime() {
    return milliseconds;
}