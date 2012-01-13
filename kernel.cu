// TODO: uredi ove includove

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "utilities.h"
#include "typedefs.h"
#include "Protein.h"
#include "Molecule.h"
#include "Info.h"

void smithWatermanCuda(Protein&, Protein&);

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

__device__  ResultType GetResult(ResultType *R, int n, int m, int i, int j) {
  if (i < -1 || j < -1 || i >= n || j >= m) return ResultType(-1.0, 0);
  return R[(i+1)*(m+1) + j+1];
}

__device__ void SetResult(ResultType *R, int n, int m, int i, int j,
                          ResultType val) {
  if (i < 0 || j < 0 || i >= n || j >= m) return;
  R[(i+1)*(m+1) + j+1] = val;
}

__device__ double DistCost(const SimpleMolecule &A, const SimpleMolecule &B) {
  return 150 - (sqr(A.x - B.x) + sqr(A.y - B.y) + sqr(A.z - B.z));
}

__global__ void Step(int n1, SimpleMolecule *A,
                     int n2, SimpleMolecule *B,
                     int k, ResultType *R) {
  int i = k - threadIdx.x;
  int j = threadIdx.x;
  
  if (i >= n1 || j >= n2) {
    return;
  }
  ResultValue a, b, c;
  a = GetResult(R, n1, n2, i-1, j-1).value + DistCost(A[i], B[j]);
  b = GetResult(R, n1, n2, i-1, j).value + A[i].dc;
  c = GetResult(R, n1, n2, i, j-1).value + B[j].dc;
  
	if (a >= b && a >= c) {
    SetResult(R, n1, n2, i, j, ResultType(a, 1));
	}
	else if (b >= a && b >= c) {
    SetResult(R, n1, n2, i, j, ResultType(b, 2));
	}
	else if (c >= b && c >= a) {
    SetResult(R, n1, n2, i, j, ResultType(c, 3));
	}
}

__device__ __host__ RESULTTYPE::RESULTTYPE() : value(0), move(-1) {}
__device__ __host__ RESULTTYPE::RESULTTYPE(ResultValue v, int m): value(v), move(m) {}


__device__ __host__ int Protein::n() const { return n_; }

void PrintResults(ResultType *R, int n, int m) {
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= m; ++j) {
      printf("%9g", R[i*(m+1) + j].value);
    }
    printf("\n");
  }
}

SimpleMolecule* SimpleMoleculeToDevice(const Protein &p, int start, int end, bool reverse=false)  {
  SimpleMolecule* host_ptr = new SimpleMolecule[end - start];
  for (int i = start; i < end; ++i) {
    if (!reverse) {
      host_ptr[i-start].x = p[i].x();
      host_ptr[i-start].y = p[i].y();
      host_ptr[i-start].z = p[i].z();
      host_ptr[i-start].dc = p[i].deletion_cost();
    } else {
      host_ptr[i-start].x = p[p.n()-1-i].x();
      host_ptr[i-start].y = p[p.n()-1-i].y();
      host_ptr[i-start].z = p[p.n()-1-i].z();
      host_ptr[i-start].dc = p[p.n()-1-i].deletion_cost();
    }
  }
  SimpleMolecule* ret = copyArrayToDevice(host_ptr, p.n());
  delete [] host_ptr;
  return ret;
}

ResultType GetResultHost(ResultType* R, int i, int j, int n, int m) {
  if (i < -1 || j < -1 || i >= n || j >= m) return ResultType(0, -1);
  if (i == -1 || j == -1) return ResultType(0, -1);
  return R[(i+1)*(m+1) + j+1];
}

ResultType* SimpleResultsToDevice(ResultType* R, int offsetX, int offsetY,
                                   int n, int m, int countX, int countY) {
  int size = (countX+1) * (countY+1);
  ResultType* host_ptr = new ResultType[size];
  for (int i = -1; i < countX; ++i) {
    for (int j = -1; j < countY; ++j) {
      host_ptr[(i+1)*(countY+1) + j+1] = GetResultHost(R, offsetX+i, offsetY+j, n, m);
    }
  }
  ResultType* ret = copyArrayToDevice(host_ptr, size);
  delete [] host_ptr;
  return ret;
}

void SimpleResultsToHost(ResultType* R, int offsetX, int offsetY, int n, int m,
                         ResultType* R_dev, int countX, int countY) {
  int size = (countX+1) * (countY+1);
  ResultType* host_ptr = copyArrayToHost(R_dev, size);
  for (int i = 1; i <= countX; ++i) {
    for (int j = 1; j <= countY; ++j) {
      R[(i+offsetX)*(m+1) + (j+offsetY)] = host_ptr[i*(countY+1) + j];
    }
  }
  delete [] host_ptr;
}

ResultType* SimpleResultsInit(int n, int m) {
  ResultType *ret = new ResultType[(n+1) * (m+1)];
  return ret;
}


void solveOnePhase(const Protein &first, const Protein &second, int block_size,
                   ResultType *results) {
  int n = first.n();
  int m = second.n();
  SimpleMolecule *A;
  SimpleMolecule *B;
  ResultType *R;

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

      A = SimpleMoleculeToDevice(first, offsetX, endX);
      B = SimpleMoleculeToDevice(second, offsetY, endY);
      R = SimpleResultsToDevice(results, offsetX, offsetY, n, m, blockX, blockY);

		  // vrti petlju
		  for (int i = 0; i < blockX+blockY-1; ++i) {
        Step<<< 1, i+1 >>>(blockX, A, blockY, B, i, R);
        SyncCudaThreads();
		  }

      SimpleResultsToHost(results,  offsetX, offsetY, n, m, R, blockX, blockY);
    }
  }

  cudaFree(A);
  cudaFree(B);
  cudaFree(R);
}


void Reconstruct(ResultType *R, int x, int y, int n, int m,
                 const Protein &A, const Protein &B,
                 std::vector< std::pair< int, int > > &res) {
  if (x < 0 || y < 0) return;
  int move = GetResultHost(R, x, y, n, m).move;
  if (move == 1) {
    res.push_back(std::make_pair(A[x].name(), B[y].name()));
    Reconstruct(R, x-1, y-1, n, m, A, B, res);
  } else if (move == 2) {
    res.push_back(std::make_pair(A[x].name(), -1));
    Reconstruct(R, x-1, y, n, m, A, B, res);
  } else if (move == 3) {
    res.push_back(std::make_pair(-1, B[y].name()));
    Reconstruct(R, x, y-1, n, m, A, B, res);
  } 
}

void Output(const std::vector< std::pair< int, int > > &A) {
  for (int i = 0; i < (int)A.size(); ++i) {
    if (A[i].first == -1) printf("%8s", "-");
    else printf("%8d", A[i].first);
    if (A[i].second == -1) printf("%8s", "-");
    else printf("%8d", A[i].second);
    printf("\n");
  }
}

void smithWatermanCuda(Protein &first, Protein &second) {
	cudaError_t cudaStatus;

  try {
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw CudaException(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		}

    int block_size = 2;
    int n = first.n();
    int m = second.n();

    ResultType *results = SimpleResultsInit(n, m);
    solveOnePhase(first, second, block_size, results);

    int MX, MY;
    double mv = -1e100;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (GetResultHost(results, i, j, n, m).value > mv) {
          MX = i;
          MY = j;
          mv = GetResultHost(results, i, j, n, m).value;
        }
      }
    }
    
    Protein first_r(first);
    first_r.Resize(MX+1);
    first_r.Reverse();
    Protein second_r(second);
    second_r.Resize(MY+1);
    second_r.Reverse();

    ResultType *results2 = SimpleResultsInit(MX+1, MY+1);
    solveOnePhase(first_r, second_r, block_size, results2);

    int mx, my;
    mv = -1e100;
    for (int i = 0; i < MX+1; ++i) {
      for (int j = 0; j < MY+1; ++j) {
        if (GetResultHost(results2, i, j, MX+1, MY+1).value > mv) {
          mx = i;
          my = j;
          mv = GetResultHost(results2, i, j, MX+1, MY+1).value;
        }
      }
    }
    
    int top = MX-mx;
    int left = MY-my;
    int bottom = MX;
    int right = MY;

    printf("Najbolje rjesenje mi je od (%d,%d) do (%d,%d)\n", top, left, bottom, right);
    std::vector< std::pair< int, int > > solution;
    Reconstruct(results2, mx, my, MX+1, MY+1, first_r, second_r, solution);
    Output(solution);

    delete [] results;
  } catch (const Exception &ex) {
		ex.print();
	}

}


void SaveResults(ResultType *R_dev, int n, int m, ResultType *R_host,
                 int offsetX, int offsetY, int N, int M) {
    ResultType *R_h = copyArrayToHost(R_dev, (n+1) * (m+1));
    for (int i = 1; i <= n; ++i) {
      for (int j = 1; j <= m; ++j) {
        R_host[(offsetX+i-1)*M + (offsetY+j-1)] = R_h[i*(m+1) + j];
      }
    }
}


Protein Protein::createCopyOnCuda() const {
  return Protein(n_, copyArrayToDevice(molecules_, n_), true);
}





////////////////////////////////////
// Implementacije kernel funkcija //
////////////////////////////////////
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

