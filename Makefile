hamming: main.cpp cudaFunctionality.cu CudaTimer.cu
	nvcc main.cpp cudaFunctionality.cu CudaTimer.cu -o hamming --compiler-options -Wall

sm_50: main.cpp cudaFunctionality.cu CudaTimer.cu
	nvcc -arch=sm_50 main.cpp cudaFunctionality.cu CudaTimer.cu -o hamming --compiler-options -Wall

clean:
	rm hamming

.PHONY:
	clean