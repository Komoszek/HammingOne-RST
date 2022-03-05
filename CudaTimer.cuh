#ifndef _CUDATIMER_CUH_
#define _CUDATIMER_CUH_

struct PrivateTimingCUDA;

class CudaTimer {
    private:
        PrivateTimingCUDA *timings;
        float milliseconds;

    public:
        CudaTimer();

        void Start();

        void Stop();

        float GetElapsedTime();

        ~CudaTimer();
};

#endif