#pragma once

#include "Molecule.h"
#include "Protein.h"
#include "typedefs.h"

using namespace nbrahle;

template< typename _T > _T power(const _T &x, const int &n) {
  if (n == 0) return 1;
  if (n == 1) return x;
  if (n & 1) return power(x, n-1) * x;
  _T tmp = power(x, n/2);
  return tmp * tmp;
}


/*!
 * \brief Returns minimum.
 */
template< typename T>
inline __device__ T min(const T &a, const T &b) { return a < b ? a : b; }


/*!
 * \brief Returns maximum.
 */
template< typename T>
inline __device__ T max(const T &a, const T &b) { return a > b ? a : b; }


/*!
 * \brief Squares the given number.
 */
template <typename T> __device__ T sqr(const T& A) {
  return A*A;
}


/*!
 * \brief Syncronizes all CUDA threads.
 */
void SyncCudaThreads() {
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		throw CudaException(cudaStatus, "cudaDeviceSynchronize je vratila pogresku nakon lansiranja kernela");
	}
}


/*!
 * \brief Helper getter function that makes an array behave like a matrix.
 */
__device__  ResultType GetResult(ResultType *R, int n, int m, int i, int j) {
  if (i < -1 || j < -1 || i >= n || j >= m) return ResultType(-1.0, 0);
  return R[(i+1)*(m+1) + j+1];
}


/*!
 * \brief Helper setter function that makes an array behave like a matrix.
 */
__device__ void SetResult(ResultType *R, int n, int m, int i, int j,
                          ResultType val) {
  if (i < 0 || j < 0 || i >= n || j >= m) return;
  R[(i+1)*(m+1) + j+1] = val;
}


/*!
 * \brief Gets the index of the diagonal a given cell (i,j) is in.
 *        Matrix dimensions are nxm.
 */
inline __device__ int getIndex(int i, int j, int n, int m) {
  if (i < 0 || j < 0) return n+m;
  int d = i+j;
  if (d < m) return i;
  return i+m-d-1;
}


/*!
 * \brief Finds the number of elements on the d-th diagonal of 
 *        a rectangle with sides n and m. 
 */
int elementsOnDiagonal(int n, int m, int d) {
  return std::min(std::min(n, m), std::min(d+1, n+m-1-d));
}


/*!
 * \brief Finds the maximum number of elements a diagonal of
 *        a rectangle with sides n and m has.
 */
int maxElemenatsOnDiagonal(int n, int m) {
  return std::min(n, m);
}


/*!
 * \brief Copies simple molecule to device.
 */
SimpleMolecule* SimpleMoleculeToDevice(const nbrahle::Protein &p, int start, int end, bool reverse=false)  {
  static int size = end - start;
  static SimpleMolecule* host_ptr = new SimpleMolecule[size];
  if (end-start > size) {
    size = end-start;
    host_ptr = new SimpleMolecule[size];
  }
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
  SimpleMolecule* ret = copyArrayToDevice(host_ptr, end-start);
  return ret;
}


/*!
 * \brief Prints results to stdout.
 */
void PrintResults(ResultType *R, int n, int m) {
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= m; ++j) {
      printf("%9g", R[i*(m+1) + j].value);
    }
    printf("\n");
  }
}


/*!
 * \brief Helper getter function on host that makes an array behave like a matrix.
 */
ResultType GetResultHost(ResultType* R, int i, int j, int n, int m) {
  if (i < -1 || j < -1 || i >= n || j >= m) return ResultType(0, -1);
  if (i == -1 || j == -1) return ResultType(0, -1);
  return R[(i+1)*(m+1) + j+1];
}


/*!
 * \brief Copies SimpleResults to device.
 */
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


/*!
 * \brief Copies SimpleResults to host.
 */
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


/*!
 * \brief Initalizes SimpleResults.
 */
ResultType* SimpleResultsInit(int n, int m) {
  ResultType *ret = new ResultType[(n+1) * (m+1)];
  return ret;
}


/*!
 * \brief Outputs allignment.
 */
void OutputAllignment(const std::vector< std::pair< int, int > > &A) {
  for (int i = 0; i < (int)A.size(); ++i) {
    if (A[i].first == -1) printf("%8s", "-");
    else printf("%8d", A[i].first);
    if (A[i].second == -1) printf("%8s", "-");
    else printf("%8d", A[i].second);
    printf("\n");
  }
}


/*!
 * \brief Finds the maximum element in SimpleResult.
 */
void FindMaxResult(double &max_value, int &mx, int &my, ResultType *results, int n, int m) {
  max_value = -1e100;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (GetResultHost(results, i, j, n, m).value > max_value) {
        mx = i;
        my = j;
        max_value = GetResultHost(results, i, j, n, m).value;
      }
    }
  }
}


/*!
 * \brief Saves SimpleResults.
 */
void SaveResults(ResultType *R_dev, int n, int m, ResultType *R_host,
                 int offsetX, int offsetY, int N, int M) {
    ResultType *R_h = copyArrayToHost(R_dev, (n+1) * (m+1));
    for (int i = 1; i <= n; ++i) {
      for (int j = 1; j <= m; ++j) {
        R_host[(offsetX+i-1)*M + (offsetY+j-1)] = R_h[i*(m+1) + j];
      }
    }
}


/*!
 * \brief Allocates an array on the device.
 */
template <typename T> void allocArrayOnDevice(T **ptr, int size) {
	cudaError_t cudaStatus = cudaMalloc((void**)ptr, sizeof(T)*size);
	if (cudaStatus != cudaSuccess) {
		throw Exception("nisam uspio alocirati polje na uredjaju");
	}
}


/*!
 * \brief Copies an array to device.
 */
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


/*!
 * \brief Copies array to host.
 */
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


/*!
 * \brief Copies array to host.
 */
template <typename T> void copyArrayToHost(T &host_ptr, const T dev_ptr, int size) {
	cudaError_t cudaStatus = cudaMemcpy(host_ptr, dev_ptr, sizeof(*dev_ptr)*size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		throw CudaException(cudaStatus, "Nisam uspio vratiti rezultat na domacina");
	}
}


///////////////////////////////////
// Class method implementations. //
///////////////////////////////////

nbrahle::Protein nbrahle::Protein::createCopyOnCuda() const {
  return nbrahle::Protein(n_, copyArrayToDevice(molecules_, n_), true);
}

double Molecule::x() const { return x_; }
double Molecule::y() const { return y_; }
double Molecule::z() const { return z_; }
 double Molecule::deletion_cost() const { return deletion_cost_; }

__device__ __host__ Molecule& nbrahle::Protein::operator[](int i) { return molecules_[i]; }
__device__ __host__ Molecule& nbrahle::Protein::operator[](int i) const { return molecules_[i]; }

__device__ __host__ RESULTTYPE::RESULTTYPE() : value(0), move(-1) {}
__device__ __host__ RESULTTYPE::RESULTTYPE(ResultValue v, int m): value(v), move(m) {}

__device__ __host__ int nbrahle::Protein::n() const { return n_; }

double* nbrahle::Protein::CopyXToDevice() const {
	double* host_ptr = new double[n_];
  for (int i = 0; i < n_; ++i) {
    host_ptr[i] = molecules_[i].x();
  }
  double* ret = copyArrayToDevice(host_ptr, n_);
  delete [] host_ptr;
  return ret;
}

double* nbrahle::Protein::CopyYToDevice() const {
	double* host_ptr = new double[n_];
  for (int i = 0; i < n_; ++i) {
    host_ptr[i] = molecules_[i].y();
  }
  double* ret = copyArrayToDevice(host_ptr, n_);
  delete [] host_ptr;
  return ret;
}

double* nbrahle::Protein::CopyZToDevice() const {
	double* host_ptr = new double[n_];
  for (int i = 0; i < n_; ++i) {
    host_ptr[i] = molecules_[i].z();
  }
  double* ret = copyArrayToDevice(host_ptr, n_);
  delete [] host_ptr;
  return ret;
}

double* nbrahle::Protein::CopyDCToDevice() const {
	double* host_ptr = new double[n_];
  for (int i = 0; i < n_; ++i) {
    host_ptr[i] = molecules_[i].deletion_cost();
  }
  double* ret = copyArrayToDevice(host_ptr, n_);
  delete [] host_ptr;
  return ret;
}

