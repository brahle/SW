#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "utilities.h"
#include "typedefs.h"
#include "Protein.h"
#include "Molecule.h"
#include "rotiraj.h"

double smithWatermanCuda(Protein&, Protein&, bool, bool);

std::vector< Point3D > getNeighbour(const std::vector< Point3D > &points) {
  std::vector< Point3D > ret;
  double dx = 5 - rand() % 11;
  double dy = 5 - rand() % 11;
  double dz = 5 - rand() % 11;
  double thetaX = 0.1 - 0.2*rand()/RAND_MAX;
  double thetaY = 0.1 - 0.2*rand()/RAND_MAX;
  double thetaZ = 0.1 - 0.2*rand()/RAND_MAX;
  RotationMatrix rotacija = createRotationMatrix(thetaX, thetaY, thetaZ);
  Point3D pomak = dajPomak(dx, dy, dz);
  for (int i = 0; i < (int)points.size(); ++i) {
    Point3D pomaknuta_tocka = rotacija*points[i] + pomak;
    ret.push_back(pomaknuta_tocka);
  }
  return ret;
}

std::vector< Point3D > original;

double getEnergy(const std::vector< Point3D > &other) {
  Protein p1(original), p2(other);
  return -smithWatermanCuda(p1, p2, true, false);
}

template< typename StateType, typename EnergyFunction, typename NextFunction >
StateType annealing(const StateType &original, EnergyFunction getEnergy,
                    NextFunction neighbour, int maxEvolution,
                    double acceptable_energy) {
  StateType current = original, next;
  double energy_current = getEnergy(original), energy_next;
  StateType best = current;
  double energy_best = energy_current;

  for (int i = 0; i < maxEvolution; ++i) {
    if (energy_best <= acceptable_energy) break;
    printf("Iteracija %d!\n", i+1);
    double T = temperature(i);
    next = neighbour(current);
    energy_next = getEnergy(next);
    printf("Najbolja energija = %g; Trenutna energija = %g\n", energy_best, energy_next);

    if (P(energy_current, energy_next, T)*RAND_MAX > rand()) {
      current = next;
      energy_current = energy_next;
    }
    if (energy_current < energy_best) {
      best = current;
      energy_best = energy_current;
    }
  }
  
  printf("Energija = %g\n", energy_best);
  return best;
}

int main()
{
  // Ucitaj podatke
  FILE *f = fopen("1a0iA.pdb", "r");
  char buff[1024];
  std::vector< Point3D > points, points2;
  printf("Ucitavam podatke...\n");
  while (fgets(buff, sizeof(buff), f)) {
    double x, y, z;
    if (sscanf(buff, "ATOM %*d CA %*s %*s %*d %lf %lf %lf", &x, &y, &z)==3) {
      points.push_back(Point3D(x,y,z));
    }
  }
  
  // Pripremi ostale podatke
  printf("Rotiram ulazni niz...\n");
  points2 = getNeighbour(points);
  for (int i = 0; i < 10; ++i) {
    points2 = getNeighbour(points2);
  }
  original = points;
  
  double start = clock();
  // Izracunaj rezultat
  cudaError_t cudaStatus;
  std::vector< Point3D > result = annealing(points2, getEnergy, getNeighbour, 5, -1e100);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
  	system("pause");
    return 1;
  }

  double end = clock();
  printf("Vrijeme = %.2lfs\n", (end-start)/CLOCKS_PER_SEC);
  // system("pause");
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

// TODO: fix this
__global__ void Step(int offsetX, int blockX, int n1, SimpleMolecule *A,
                     int offsetY, int blockY, int n2, SimpleMolecule *B,
                     int k, ResultType *R) {
  int i = k - threadIdx.x;
  int j = threadIdx.x;
  
  if (i >= blockX || j >= blockY) {
    return;
  }
  ResultValue a, b, c;
  a = GetResult(R, n1, n2, i-1, offsetY+j-1).value + DistCost(A[offsetX+i], B[offsetY+j]);
  b = GetResult(R, n1, n2, i-1, offsetY+j).value + A[offsetX+i].dc;
  c = GetResult(R, n1, n2, i, offsetY+j-1).value + B[offsetY+j].dc;
  
	if (a >= b && a >= c) {
    SetResult(R, n1, n2, i, offsetY+j, ResultType(a, 1));
	}
	else if (b >= a && b >= c) {
    SetResult(R, n1, n2, i, offsetY+j, ResultType(b, 2));
	}
	else if (c >= b && c >= a) {
    SetResult(R, n1, n2, i, offsetY+j, ResultType(c, 3));
	}
}


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
  SimpleMolecule* ret = copyArrayToDevice(host_ptr, end-start);
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
                   ResultType *results, bool silent=false) {
  int n = first.n();
  int m = second.n();
  SimpleMolecule *A;
  SimpleMolecule *B;
  ResultType *R;
  A = SimpleMoleculeToDevice(first, 0, n);
  B = SimpleMoleculeToDevice(second, 0, n);

  for (int offsetX = 0; offsetX < n; offsetX += block_size) {
    int blockX = block_size;
    int endX = offsetX + block_size;
    if (offsetX + block_size >= n) {
      blockX = n - offsetX;
      endX = n;
    }
    R = SimpleResultsToDevice(results, offsetX, 0, n, m, blockX, m);

    for (int offsetY = 0; offsetY < m; offsetY += block_size) {
      int blockY = block_size;
      int endY = offsetY + block_size;
      if (offsetY + block_size >= m) {
        blockY = m - offsetY;
        endY = m;
      }

      if (!silent) {
        printf("Rjesavam blok od (%d,%d) do (%d,%d)!\n", offsetX, offsetY, endX, endY);
      }

		  // vrti petlju
		  for (int i = 0; i < blockX+blockY-1; ++i) {
        Step<<< 1, i+1 >>>(offsetX, blockX, blockX, A, offsetY, blockY, m, B, i, R);
        SyncCudaThreads();
		  }
    }
    SimpleResultsToHost(results,  offsetX, 0, n, m, R, blockX, m);
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

double smithWatermanCuda(Protein &first, Protein &second, bool silent=false, bool reconstruct=true) {
	cudaError_t cudaStatus;
  double res = 0;

  try {
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw CudaException(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		}

    int block_size = 400;
    int n = first.n();
    int m = second.n();

    ResultType *results = SimpleResultsInit(n, m);
    solveOnePhase(first, second, block_size, results, silent);

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
    solveOnePhase(first_r, second_r, block_size, results2, silent);

    int mx, my;
    mv = -1e100;
    for (int i = 0; i < MX+1; ++i) {
      for (int j = 0; j < MY+1; ++j) {
        if (GetResultHost(results2, i, j, MX+1, MY+1).value > mv) {
          mx = i;
          my = j;
          mv = GetResultHost(results2, i, j, MX+1, MY+1).value;
          res = mv;
        }
      }
    }
    
    int top = MX-mx;
    int left = MY-my;
    int bottom = MX;
    int right = MY;
    
    if (!silent) {
      printf("Najbolje rjesenje mi je od (%d,%d) do (%d,%d)\n", top, left, bottom, right);
    }
    if (reconstruct) {
      std::vector< std::pair< int, int > > solution;
      Reconstruct(results2, mx, my, MX+1, MY+1, first_r, second_r, solution);
      if (!silent) {
        Output(solution);
      }
    }

    delete [] results;
  } catch (const Exception &ex) {
		ex.print();
	}

  return res;
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



///////////////////////////////////////////////
// Implementacije pomocnih i kernel funkcija //
///////////////////////////////////////////////
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

__device__ __host__ RESULTTYPE::RESULTTYPE() : value(0), move(-1) {}
__device__ __host__ RESULTTYPE::RESULTTYPE(ResultValue v, int m): value(v), move(m) {}

__device__ __host__ int Protein::n() const { return n_; }


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

