// TODO: uredi ove includove

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <iostream>
#include <fstream>

#include "utilities.h"
#include "typedefs.h"
#include "Protein.h"
#include "Molecule.h"
#include "Info.h"
#include "Results.h"

void smithWatermanCuda(const Protein&, const Protein&);

int main()
{
  // TODO: ovdje nesto fali
  cudaError_t cudaStatus;
  std::ifstream input("input.txt", std::ifstream::in);
  Protein p1, p2;
  input >> p1 >> p2;
  
  smithWatermanCuda(p1, p2);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
  	system("pause");
    return 1;
  }

	system("pause");
  return 0;
}


void SyncCudaThreads() {
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		throw CudaException(cudaStatus, "cudaDeviceSynchronize je vratila pogresku nakon lansiranja kernela");
	}
}


template <typename T> __device__ T sqr(const T& A) {
  return A*A;
}

struct SimpleMolecule {
  double x, y, z, dc;
};

__device__  ResultsType GetResult(ResultsType *R, int n, int m, int i, int j) {
  if (i < -1 || j < -1 || i >= n || j >= m) return 0;
  return R[(i+1)*(m+1) + j+1];
}

__device__ void SetResult(ResultsType *R, int n, int m, int i, int j,
                          ResultsType val) {
  if (i < 0 || j < 0 || i >= n || j >= m) return;
  R[(i+1)*(m+1) + j+1] = val;
}

__device__ double DistCost(const SimpleMolecule &A, const SimpleMolecule &B) {
  return sqr(A.x - B.x) + sqr(A.y - B.y) + sqr(A.z - B.z);
}

__global__ void Step(int n1, SimpleMolecule *A,
                     int n2, SimpleMolecule *B,
                     int k, ResultsType *R) {
  int i = k - threadIdx.x;
  int j = threadIdx.x;
  
  if (i >= n1 || j >= n2) {
    return;
  }
  ResultsType a, b, c;
  a = GetResult(R, n1, n2, i-1, j-1) + DistCost(A[i], B[j]);
  b = GetResult(R, n1, n2, i-1, j) + A[i].dc;
  c = GetResult(R, n1, n2, i, j-1) + B[j].dc;
  
	if (a >= b && a >= c) {
    SetResult(R, n1, n2, i, j, a);
	}
	else if (b >= a && b >= c) {
    SetResult(R, n1, n2, i, j, b);
	}
	else if (c >= b && c >= a) {
    SetResult(R, n1, n2, i, j, c);
	}
}


__device__ __host__ int Protein::n() const { return n_; }

void PrintResults(ResultsType *R, int n, int m) {
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= m; ++j) {
      printf("%8g", R[i*(m+1) + j]);
    }
    printf("\n");
  }
}

SimpleMolecule* SimpleMoleculeToDevice(const Protein &p, int start, int end)  {
  SimpleMolecule* host_ptr = new SimpleMolecule[end - start];
  for (int i = start; i < end; ++i) {
    host_ptr[i-start].x = p[i].x();
    host_ptr[i-start].y = p[i].y();
    host_ptr[i-start].z = p[i].z();
    host_ptr[i-start].dc = p[i].deletion_cost();
  }
  SimpleMolecule* ret = copyArrayToDevice(host_ptr, p.n());
  delete [] host_ptr;
  return ret;
}

ResultsType GetResultHost(ResultsType* R, int i, int j, int n, int m) {
  if (i < -1 || j < -1 || i >= n || j >= m) return 0;
  if (i == -1 || j == -1) return 0;
  return R[(i+1)*(m+1) + j+1];
}

ResultsType* SimpleResultsToDevice(ResultsType* R, int offsetX, int offsetY,
                                   int n, int m, int countX, int countY) {
  int size = (countX+1) * (countY+1);
  ResultsType* host_ptr = new ResultsType[size];
  for (int i = -1; i < countX; ++i) {
    for (int j = -1; j < countY; ++j) {
      host_ptr[(i+1)*(countY+1) + j+1] = GetResultHost(R, offsetX+i, offsetY+j, n, m);
    }
  }
  ResultsType* ret = copyArrayToDevice(host_ptr, size);
  delete [] host_ptr;
  return ret;
}

void SimpleResultsToHost(ResultsType* R, int offsetX, int offsetY, int n, int m,
                         ResultsType* R_dev, int countX, int countY) {
  int size = (countX+1) * (countY+1);
  ResultsType* host_ptr = copyArrayToHost(R_dev, size);
  for (int i = 1; i <= countX; ++i) {
    for (int j = 1; j <= countY; ++j) {
      R[(i+offsetX)*(m+1) + (j+offsetY)] = host_ptr[i*(countY+1) + j];
    }
  }
  delete [] host_ptr;
}

ResultsType* SimpleResultsInit(int n, int m) {
  ResultsType *ret = new ResultsType[(n+1) * (m+1)];
  memset(ret, 0, sizeof(ResultsType) * (n+1) * (m+1));
  return ret;
}

void smithWatermanCuda(const Protein &prvi, const Protein &drugi) {
	cudaError_t cudaStatus;

	int n = prvi.n();
  int m = drugi.n();
  ResultsType *results = SimpleResultsInit(n, m);
  SimpleMolecule *A;
  SimpleMolecule *B;
  ResultsType *R;
  
  try {
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw CudaException(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		}

    int block_size = 2;

    for (int offsetX = 0; offsetX < n; offsetX += block_size) {
      for (int offsetY = 0; offsetY < m; offsetY += block_size) {
        int blockX = block_size;
        int blockY = block_size;
        int endX = offsetX + block_size;
        int endY = offsetY + block_size;
        if (offsetX + block_size >= n) {
          blockX = n - offsetX;
          endX = n;
        }
        if (offsetY + block_size >= m) {
          blockY = m - offsetY;
          endY = m;
        }
        printf("Rjesavam blok od (%d,%d) do (%d,%d)!\n", offsetX, offsetY, endX, endY);

        A = SimpleMoleculeToDevice(prvi, offsetX, endX);
        B = SimpleMoleculeToDevice(drugi, offsetY, endY);
        R = SimpleResultsToDevice(results, offsetX, offsetY, n, m, blockX, blockY);

		    // vrti petlju
		    for (int i = 0; i < blockX+blockY-1; ++i) {
          Step<<< 1, i+1 >>>(blockX, A, blockY, B, i, R);
          SyncCudaThreads();
		    }

        SimpleResultsToHost(results,  offsetX, offsetY, n, m, R, blockX, blockY);
      }
    }
    PrintResults(results, n, m);
	} catch (const Exception &ex) {
		ex.print();
	}

  cudaFree(A);
  cudaFree(B);
  cudaFree(R);
  delete [] results;
}


void SaveResults(ResultsType *R_dev, int n, int m, ResultsType *R_host,
                 int offsetX, int offsetY, int N, int M) {
    ResultsType *R_h = copyArrayToHost(R_dev, (n+1) * (m+1));
    for (int i = 1; i <= n; ++i) {
      for (int j = 1; j <= m; ++j) {
        R_host[(offsetX+i-1)*M + (offsetY+j-1)] = R_h[i*(m+1) + j];
      }
    }
}


Protein Protein::createCopyOnCuda() const {
  return Protein(n_, copyArrayToDevice(molecules_, n_), true);
}

Results Results::CreateCopyOnDevice() {
  return Results(
    n_,
    m_,
    copyArrayToDevice(results_, n_*m_),
    copyArrayToDevice(previous_row_, m_),
    copyArrayToDevice(previous_column_, n_),
    special_,
    true
  );
}


Results Results::CreateCopyOnHost() {
  return Results(
    n_,
    m_,
    copyArrayToHost(results_, n_*m_),
    copyArrayToHost(previous_row_, m_),
    copyArrayToHost(previous_column_, n_),
    special_,
    false
  );
}





////////////////////////////////////
// Implementacije kernel funkcija //
////////////////////////////////////
ResultsType Results::GetResult(int i, int j) const {
  if (i < -1 || j < -1 || i >= n_ || j >= m_) return 0;
  if (i == -1) {
    if (j == -1) return special_;
    return previous_column_[j];
  }
  if (j == -1) {
    return previous_row_[i];
  }
  return results_[i*m_ + j];
}

__device__ __host__ void Results::SetResult(int i, int j, ResultsType value) {
  results_[i*m_ + j] = value;
}

double Molecule::x() const { 
  return x_;
}
double Molecule::y() const {
  return y_;
}
double Molecule::z() const {
  return z_;
}
 double Molecule::deletion_cost() const {
  return deletion_cost_;
}

__device__ __host__ Molecule& Protein::operator[](int i) { return molecules_[i]; }
__device__ __host__ Molecule& Protein::operator[](int i) const { return molecules_[i]; }

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

double* Protein::CopyXToDevice() const {
	double* host_ptr = new double[n_];
  for (int i = 0; i < n_; ++i) {
    host_ptr[i] = molecules_[i].x();
  }
  double* ret = copyArrayToDevice(host_ptr, n_);
  delete [] host_ptr;
  return ret;
}

double* Protein::CopyYToDevice() const {
	double* host_ptr = new double[n_];
  for (int i = 0; i < n_; ++i) {
    host_ptr[i] = molecules_[i].y();
  }
  double* ret = copyArrayToDevice(host_ptr, n_);
  delete [] host_ptr;
  return ret;
}

double* Protein::CopyZToDevice() const {
	double* host_ptr = new double[n_];
  for (int i = 0; i < n_; ++i) {
    host_ptr[i] = molecules_[i].z();
  }
  double* ret = copyArrayToDevice(host_ptr, n_);
  delete [] host_ptr;
  return ret;
}

double* Protein::CopyDCToDevice() const {
	double* host_ptr = new double[n_];
  for (int i = 0; i < n_; ++i) {
    host_ptr[i] = molecules_[i].deletion_cost();
  }
  double* ret = copyArrayToDevice(host_ptr, n_);
  delete [] host_ptr;
  return ret;
}

