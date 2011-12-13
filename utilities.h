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
public:
	Exception(const std::string &_message="") : message(_message) {}

	void print() const {
		printf("Exception: %s!\n", message.c_str());
	}
};

class CudaException : public Exception {
protected:
	std::string message;
	cudaError_t error;
public:
	CudaException(const cudaError_t &_error, const std::string &_message="") : error(_error), Exception(_message) {}

	void print() const {
		printf("Cuda Exception %d: %s!\n", error, message.c_str());
	}
};

class CudaAllocationException : public CudaException {
public:
	CudaAllocationException(const cudaError_t &_error, const std::string &_message="") : CudaException(_error, _message) {}

	void print() const {
		printf("Allocation Exception %d: %s!\n", error, message.c_str());
	}
};

class CudaMemcpyException : public CudaException {
public:
	CudaMemcpyException(const cudaError_t &_error, const std::string &_message="") : CudaException(_error, _message) {}

	void print() const {
		printf("Memcpy Exception %d: %s!\n", error, message.c_str());
	}
};

class CudaSyncException : public CudaException {
public:
	CudaSyncException(const cudaError_t &_error, const std::string &_message="") : CudaException(_error, _message) {}

	void print() const {
		printf("Sync Exception %d: %s!\n", error, message.c_str());
	}
};



//////////////////////////////////////////////////
// Funkcije za kopiranje na karticu i s kartice //
//////////////////////////////////////////////////
template <typename T> T* copyArrayToDevice(const T *ptr, int size) {
	T* dev_ptr;
	cudaError_t cudaStatus = cudaMalloc((void**)&dev_ptr, sizeof(T)*size);
	if (cudaStatus != cudaSuccess) {
		throw Exception("nisam uspio alocirati polje na uredjaju");
	}
	cudaStatus = cudaMemcpy(dev_ptr, ptr, sizeof(T)*size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		throw CudaMemcpyException(cudaStatus, "nisam uspio iskopirati polje na uredjaj");
	}
	return dev_ptr;
}


template <typename T> T* copyArrayToHost(const T *ptr, int size) {
	T* host_ptr = new T[size];
  if (host_ptr == 0) {
    throw Exception("nisam uspio alocirati polje na domacinu");
  }
	cudaError_t cudaStatus = cudaMemcpy(host_ptr, ptr, sizeof(T)*size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		throw CudaMemcpyException(cudaStatus, "nisam uspio vratiti rezultat na domacina");
	}
	return host_ptr;
}

template <typename T> void copyArrayToHost(T &host_ptr, const T dev_ptr, int size) {
	cudaError_t cudaStatus = cudaMemcpy(host_ptr, dev_ptr, sizeof(*dev_ptr)*size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		throw CudaException(cudaStatus, "Nisam uspio vratiti rezultat na domacina");
	}
}


////////////////////////////////////////////
// Funkcije za sinkronizaciju svih dretvi //
////////////////////////////////////////////
void syncCudaThreads() {
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		throw CudaException(cudaStatus, "cudaDeviceSynchronize je vratila pogresku nakon lansiranja kernela");
	}
}


////////////////////////////
// Razne korisne funkcije //
////////////////////////////
template <typename T> __device__ T sqr(const T& A) {
  return A*A;
}



#endif