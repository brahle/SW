#ifndef __BRAHLE_CUDA_UTILITIES
#define __BRAHLE_CUDA_UTILITIES

#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//////////////////////////////////////
// Razne iznimke koje mozemo baciti //
//////////////////////////////////////
class Exception {
protected:
	std::string message;
	cudaError_t error;
public:
	Exception(const cudaError_t &_error, const std::string &_message="") : error(_error), message(_message) {}

	void print() const {
		printf("Exception %d: %s!\n", error, message.c_str());
	}
};

class CudaAllocationException : public Exception {
public:
	CudaAllocationException(const cudaError_t &_error, const std::string &_message="") : Exception(_error, _message) {}

	void print() const {
		printf("Allocation Exception %d: %s!\n", error, message.c_str());
	}
};

class CudaMemcpyException : public Exception {
public:
	CudaMemcpyException(const cudaError_t &_error, const std::string &_message="") : Exception(_error, _message) {}

	void print() const {
		printf("Memcpy Exception %d: %s!\n", error, message.c_str());
	}
};

class CudaSyncException : public Exception {
public:
	CudaSyncException(const cudaError_t &_error, const std::string &_message="") : Exception(_error, _message) {}

	void print() const {
		printf("Sync Exception %d: %s!\n", error, message.c_str());
	}
};



//////////////////////////////////////////////////
// Funkcije za kopiranje na karticu i s kartice //
//////////////////////////////////////////////////
template <typename T> T* kopirajArrayNaDevice(const T *ptr, int size) {
	T* dev_ptr;
	cudaStatus = cudaMalloc((void**)&dev_ptr, sizeof(T)*size);
	if (cudaStatus != cudaSuccess) {
		throw CudaAllocationException(cudaStatus, "nisam uspio alocirati polje");
	}
	cudaStatus = cudaMemcpy(dev_ptr, ptr, sizeof(T)*size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		throw CudaMemcpyException(cudaStatus, "nisam uspio iskopirati polje");
	}
	return dev_ptr;
}

template <typename T> void kopirajArrayNaHost(T &host_ptr, const T dev_ptr, int size) {
	cudaError_t cudaStatus = cudaMemcpy(host_ptr, dev_ptr, sizeof(*T)*size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		throw Exception("Nisam uspio vratiti rezultat na domacina");
	}
	return host_ptr;
}

////////////////////////////////////////////
// Funkcije za sinkronizaciju svih dretvi //
////////////////////////////////////////////
void sinkronizirajCudaDretve() {
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		throw Exception(cudaStatus, "cudaDeviceSynchronize je vratila pogresku nakon lansiranja kernela");
	}
}
#endif