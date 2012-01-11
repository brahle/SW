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


class A 
{
 public:
  __device__ int Bar(int);
 private:
  int *foo_;
};

__device__ int A::Bar(int x) {
  return foo_[x];
}


class ResultsD
{
 public:
  ResultsD(void);
//  ResultsD(int, int, ResultsTypePtr, ResultsTypePtr, ResultsTypePtr,
//          ResultsType, bool);
  ~ResultsD(void);

//  void Init(int, int);
//  void AdvanceToNewRow(const ResultsType, const ResultsTypePtr);
//  void Advance(const ResultsType, const ResultsTypePtr);
//  Results CreateCopyOnDevice();
//  Results CreateCopyOnHost();

  __device__ ResultsType GetResult(int, int) const;
//  ResultsTypePtr GetLastRow() const;
//  void CopyLastRow(const ResultsTypePtr) const;
//  void print() const;

  __device__ void SetResult(int i, int j, ResultsType value);

 public:
  int n_, m_;
  ResultsTypePtr results_;
  ResultsTypePtr previous_row_;
  ResultsTypePtr previous_column_;
  ResultsType special_; // TODO: better name
};

ResultsD::ResultsD() {}
ResultsD::~ResultsD() {}


void smithWatermanCuda(Protein, Protein);


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

  printf("dosao sam do kraja...");
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


__device__ double DistCost(const Molecule &A, const Molecule &B) {
  return sqr(A.x() - B.x()) + sqr(A.y() - B.y()) + sqr(A.z() - B.z());
}


__global__ void OneElement(Protein protein_A, Protein protein_B, int k,
                           ResultsD R, ResultsTypePtr bla) {
  int i = k - threadIdx.x;
  int j = threadIdx.x;
  
  if (i >= protein_A.n() || j >= protein_B.n()) {
    return;
  }
  const Molecule &molecule_A = protein_A[i];
  const Molecule &molecule_B = protein_B[j];

  ResultsType a, b, c;
  a = R.GetResult(i-1, j-1) + DistCost(molecule_A, molecule_B);
  b = R.GetResult(i-1, j) + molecule_A.deletion_cost();
  c = R.GetResult(i, j-1) + molecule_B.deletion_cost();
  
	if (a >= b && a >= c) {
    R.SetResult(i, j, a);
	}
	if (b >= a && b >= c) {
    R.SetResult(i, j, b);
	}
	if (c >= b && c >= a) {
    R.SetResult(i, j, c);
	}
}

__device__ __host__ int Protein::n() const { return n_; }


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

		// Alociraj prvi i drugi protein na cudi.
		Protein dev_prvi = prvi.createCopyOnCuda();
		Protein dev_drugi = drugi.createCopyOnCuda();

/*    Protein *ddev_prvi;
    cudaMalloc((void**)&ddev_prvi, sizeof(Protein));
    cudaMemcpy(ddev_prvi, &dev_prvi, sizeof(Protein), cudaMemcpyHostToDevice);*/

		// Alociraj i rezultat na cudi.
    dev_results = results.CreateCopyOnDevice();

    ResultsTypePtr bla = copyArrayToDevice(results.results_, results.n_*results.m_);
    ResultsD dev_results2;
    dev_results2.n_ = results.n_;
    dev_results2.m_ = results.m_;
    dev_results2.results_ = copyArrayToDevice(results.results_, results.n_*results.m_);
    dev_results2.previous_row_ = copyArrayToDevice(results.previous_row_, results.n_);
    dev_results2.previous_column_ = copyArrayToDevice(results.previous_column_, results.m_);
    dev_results2.special_ = results.special_;
    
/*    Results *ddev_results;
    cudaMalloc((void**)&ddev_results, sizeof(Results));
    cudaMemcpy(ddev_results, &dev_results, sizeof(Results), cudaMemcpyHostToDevice);*/

		// vrti petlju
		for (int i = 0; i < n+m-1; ++i) {
/*  		Protein dev_prvi = prvi.createCopyOnCuda();
	  	Protein dev_drugi = drugi.createCopyOnCuda(); */

      printf("Zovem kernel za %d. dijagonalu\n", i);
			OneElement<<< 1, i+1 >>>(dev_prvi, dev_drugi, i, dev_results2, bla);
			SyncCudaThreads();

		  // Vrati rezultat natrag na host.
/*      results.results_ = copyArrayToHost(dev_results2.results_, results.n_*results.m_);
      results.previous_row_ = copyArrayToHost(dev_results2.previous_row_, results.n_);
      results.previous_column_ = copyArrayToHost(dev_results2.previous_column_, results.m_);
      results.special_ = dev_results2.special_;
      results.print(); */
		}

		// Vrati rezultat natrag na host.
    results.results_ = copyArrayToHost(dev_results2.results_, results.n_*results.m_);
    results.previous_row_ = copyArrayToHost(dev_results2.previous_row_, results.n_);
    results.previous_column_ = copyArrayToHost(dev_results2.previous_column_, results.m_);
    results.special_ = dev_results.special_;
    results.print();
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

__device__ ResultsType get_result(int i, int j) {

}

__device__ ResultsType ResultsD::GetResult(int i, int j) const {
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

__device__ void ResultsD::SetResult(int i, int j, ResultsType value) {
  results_[i*m_ + j] = value;
}





__device__ ResultsType Results::GetResultD(int i, int j) const {
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

__host__ ResultsType Results::GetResultH(int i, int j) const {
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

__device__ __host__ double Molecule::x() const { 
  return x_;
}
__device__ __host__ double Molecule::y() const {
  return y_;
}
__device__ __host__ double Molecule::z() const {
  return z_;
}
__device__ __host__ double Molecule::deletion_cost() const {
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
