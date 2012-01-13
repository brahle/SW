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

void smithWatermanCuda(Protein, Protein);

int main()
{
  // TODO: ovdje nesto fali
  cudaError_t cudaStatus;
  std::ifstream input("input.txt", std::ifstream::in);
  Protein p1, p2;
  input >> p1 >> p2;
  
  smithWatermanCuda(p1, p2);
  printf("dosao sam do kraja...\n");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
  	system("pause");
    return 1;
  }

  printf("dosao sam do kraja...\n");
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

__device__ void SetResult(ResultsType *R, int n, int m, int i, int j, ResultsType val) {
  if (i < 0 || j < 0 || i >= n || j >= m) return;
  R[(i+1)*(m+1) + j+1] = val;
}

__device__ double DistCost(const SimpleMolecule &A, const SimpleMolecule &B) {
  return sqr(A.x - B.x) + sqr(A.y - B.y) + sqr(A.z - B.z);
}

__global__ void Step2(int n1, SimpleMolecule *A,
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


SimpleMolecule* SimpleMoleculeToDevice(Protein p)  {
  SimpleMolecule* host_ptr = new SimpleMolecule[p.n()];
  for (int i = 0; i < p.n(); ++i) {
    host_ptr[i].x = p[i].x();
    host_ptr[i].y = p[i].y();
    host_ptr[i].z = p[i].z();
    host_ptr[i].dc = p[i].deletion_cost();
  }
  SimpleMolecule* ret = copyArrayToDevice(host_ptr, p.n());
  delete [] host_ptr;
  return ret;
}


ResultsType* SimpleResultsToDevice(Results R) {
  int size = (R.n_+1) * (R.m_+1);
  ResultsType* host_ptr = new ResultsType[size];
  for (int i = -1; i < R.n_; ++i) {
    for (int j = -1; j < R.m_; ++j) {
      host_ptr[(i+1)*(R.m_+1) + j+1] = R.GetResult(i, j);
    }
  }
  ResultsType* ret = copyArrayToDevice(host_ptr, size);
  delete [] host_ptr;
  return ret;
}


void smithWatermanCuda(Protein prvi, Protein drugi) {
	cudaError_t cudaStatus;
  Results results;
	Results dev_results;

	int n = prvi.n();
	int m = drugi.n();
  results.Init(n, m);
  results.SetResult(0, 0, 20);
  results.print();

  try {
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw CudaException(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		}

    int n1 = prvi.n();
    SimpleMolecule *A = SimpleMoleculeToDevice(prvi);
    int n2 = drugi.n();
    SimpleMolecule *B = SimpleMoleculeToDevice(drugi);

    ResultsType *R = SimpleResultsToDevice(results);
		// vrti petlju
		for (int i = 0; i < n+m-1; ++i) {
      printf("Zovem kernel za %d. dijagonalu\n", i);

      Step2<<< 1, i+1 >>>(n1, A, n2, B, i, R);
      SyncCudaThreads();
		}

		// Vrati rezultat natrag na host.
    ResultsType *R_h = copyArrayToHost(R, (n1+1) * (n2+1));
    for (int i = 0; i <= n1; ++i) {
      for (int j = 0; j <= n2; ++j) {
        printf("%8g", R_h[i*(n2+1) + j]);
      }
      printf("\n");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(R);
	} catch (const Exception &ex) {
		ex.print();
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

