#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "consts.h"
#include "external_structs.h"
#include "Molecule.h"
#include "Protein.h"
#include "rotate.h"
#include "State.h"
#include "typedefs.h"
#include "utilities.h"

#include "annealing.cuh"
#include "find_maximum.cuh"
#include "utilities.cuh"

using namespace nbrahle;

double smithWatermanCuda(nbrahle::Protein&, nbrahle::Protein&, bool, bool);
double smithWatermanCudaFast(nbrahle::Protein&, nbrahle::Protein&);

State original;

/*!
 * \brief Transforms given state (using rotation and translation) to
 *        get a neighbouring state for simulated annealing. 
 */
State getNeighbour(const State &A) {
  State ret;
  ret.transformOther(A);
  return ret;
}

/*!
 * \brief Calculates energy of the current state by running Smith-Waterman
          algorithm on that state and the original one.
 */
double getEnergy(const State &other) {
  nbrahle::Protein p1(original.P), p2(other.P);
  return -smithWatermanCudaFast(p1, p2);
}

/*! 
 * \brief Main program. 
 */
int main() {
  cudaError_t cudaStatus;
  State prvi, drugi;
    
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		throw CudaException(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

//  prvi.ucitaj("1d0nA.pdb");
//  drugi.ucitaj("2d8bA.pdb");

  prvi.read("1a0iA.pdb");
  printf("Rotiram ulazni niz...\n");
  drugi = prvi;
  drugi.transformMyself(10, 1.5);


  original = prvi;
  
  printf("Najbolja teoretska energija: %g\n", getEnergy(original));

  double start = clock();
  // Izracunaj rezultat
  State result = annealing(drugi, getEnergy, getNeighbour, ANNEALING_STEPS, -1e100, POOL_SIZE);
  double end = clock();

  printf("Vrijeme = %.2lfs\n", (end-start)/CLOCKS_PER_SEC);

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





/*!
 * The current best guess for the rotation and translation are in
 * map->q (the rotation representeed as a 4-component quaternion; the components defined as doubles),
 * and  map->T (3 component; double); to get the rotation matrix use 
 * quat_to_R (q, R); defined in  04_geometric_match/struct_quaternion.c:30
 * Map is defined in 00_include/struct.h:190      
 */
int single_map_optimize_bb_almt (::Protein *p1, ::Protein *p2, Map * map) {
  State S1(p1), S2(p2);
  original = S1;
  State res = annealing(S2, getEnergy, getNeighbour, 100, -1e100, 20);
  /* after optimization replace map->q and map->T with the new values */

  map->T[0] = res.dx;
  map->T[1] = res.dy;
  map->T[2] = res.dz;

  RotationMatrix r = createRotationMatrix(res.thetaX, res.thetaY, res.thetaZ);
  double w, x, y, z;

  /* code from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm */
  float trace = r.P[0][0] + r.P[1][1] + r.P[2][2];
  if( trace > 0 ) {
    float s = 0.5f / sqrtf(trace+ 1.0f);
    w = 0.25f / s;
    x = ( r.P[2][1] - r.P[1][2] ) * s;
    y = ( r.P[0][2] - r.P[2][0] ) * s;
    z = ( r.P[1][0] - r.P[0][1] ) * s;
  } else {
    if ( r.P[0][0] > r.P[1][1] && r.P[0][0] > r.P[2][2] ) {
      float s = 2.0f * sqrtf( 1.0f + r.P[0][0] - r.P[1][1] - r.P[2][2]);
      w = (r.P[2][1] - r.P[1][2] ) / s;
      x = 0.25f * s;
      y = (r.P[0][1] + r.P[1][0] ) / s;
      z = (r.P[0][2] + r.P[2][0] ) / s;
    } else if (r.P[1][1] > r.P[2][2]) {
      float s = 2.0f * sqrtf( 1.0f + r.P[1][1] - r.P[0][0] - r.P[2][2]);
      w = (r.P[0][2] - r.P[2][0] ) / s;
      x = (r.P[0][1] + r.P[1][0] ) / s;
      y = 0.25f * s;
      z = (r.P[1][2] + r.P[2][1] ) / s;
    } else {
      float s = 2.0f * sqrtf( 1.0f + r.P[2][2] - r.P[0][0] - r.P[1][1] );
      w = (r.P[1][0] - r.P[0][1] ) / s;
      x = (r.P[0][2] + r.P[2][0] ) / s;
      y = (r.P[1][2] + r.P[2][1] ) / s;
      z = 0.25f * s;
    }
  }

  map->q[0] = w;
  map->q[1] = x;
  map->q[2] = y;
  map->q[3] = z;
  return 0;
}




/*!
 * \brief Gets distance cost between two molceules.
 */
inline __device__ double DistCost(const SimpleMolecule &A, const SimpleMolecule &B) {
  return 1000 - exp((sqr(A.x - B.x) + sqr(A.y - B.y) + sqr(A.z - B.z))/100);
}


/*!
 * \brief Calculates one cell of the result, works better for bigger proteins.
 */
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
  
  if (0 > a && 0 > b && 0 > c) {
    SetResult(R, n1, n2, i, offsetY+j, ResultType(0, 0));
  }
	else if (a >= b && a >= c) {
    SetResult(R, n1, n2, i, offsetY+j, ResultType(a, 1));
	}
	else if (b >= a && b >= c) {
    SetResult(R, n1, n2, i, offsetY+j, ResultType(b, 2));
	}
	else if (c >= b && c >= a) {
    SetResult(R, n1, n2, i, offsetY+j, ResultType(c, 3));
	}
}


/*!
 * \brief Calculates one cell of the result, optimized for smaller
 *        proteins.
 */
__global__ void StepFast(double *H, double *Ha, double *Hb,
                         double *H1, double *H1a, double *H1b,
                         double *H2, double *H2a, double *H2b,
                         SimpleMolecule *A, SimpleMolecule *B,
                         int n, int m, int d, int T,
                         double D, double C) {
  int offset = min(0, m-d-1);
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= T) return;
  int x = idx - offset;
  int y = d - idx + offset;
  int ja = idx;
  if (x == 0 || y == 0) {
    H[ja] = 0.0;
    Ha[ja] = 0.0;
    Hb[ja] = 0.0;
    return;
  }

  int gore = getIndex(x-1, y, n, m);
  int lijevo = getIndex(x, y-1, n, m);
  int oba = getIndex(x-1, y-1, n, m);

  double a = max(0.0, H2[oba] + DistCost(A[x], B[y]));
  double b = H1a[gore] + C;
  double c = H1b[lijevo] + C;

  H[ja] = max(a, max(b, c));

  b = H1a[gore] + D;
  Ha[ja] = max(a, max(b, c));

  b = H1a[gore] + C;
  c = H1b[lijevo] + D;
  Hb[ja] = max(a, max(b, c));
}

/*!
 * \brief Smith-Waterman algorithm, optimized for smaller proteins. Does not do reconstruction.
 */
double smithWatermanCudaFast(nbrahle::Protein &first, nbrahle::Protein &second) {
	cudaError_t cudaStatus;
  double res = -1e100;

  try {
    int n = first.n();
    int m = second.n();
    int l = maxElemenatsOnDiagonal(n, m)+1;
    double *H[3], *Ha[3], *Hb[3];
    int trenutni = 0, prosli = 1, pretprosli = 2;
    register int i, d;

    SimpleMolecule *A_device = SimpleMoleculeToDevice(first, 0, n, false);
    SimpleMolecule *B_device = SimpleMoleculeToDevice(second, 0, m, false);
    
    double *h = new double[l];
    for (i = 0; i < l; ++i) h[i] = 0.0;
    for (i = 0; i < 3; ++i) {
      H[i] = copyArrayToDevice(h, l);
      Ha[i] = copyArrayToDevice(h, l);
      Hb[i] = copyArrayToDevice(h, l);
    }

    for (d = 0; d < n+m-1; ++d) {
      int k = elementsOnDiagonal(n, m, d);
      int blocks = (k+511)/512;
      int threads = std::min(k, 512);

      StepFast<<< blocks, threads >>>( H[trenutni],   Ha[trenutni],   Hb[trenutni],
                                       H[prosli],     Ha[prosli],     Hb[prosli],
                                       H[pretprosli], Ha[pretprosli], Hb[pretprosli],
                                       A_device, B_device, n, m, d, k,
                                       GAP_START, GAP_CONTINUE);
      //SyncCudaThreads();
      if (k < 4192) {
        cudaStatus = cudaMemcpy(h, H[trenutni], sizeof(double)*k, cudaMemcpyDeviceToHost);
     		if (cudaStatus != cudaSuccess) {
	    		throw CudaException(cudaStatus, "cudaMemcmpy failed! Nema dovoljno memorije.");
		    }

        for (i = 0; i < k; ++i) {
          if (res < h[i]) {
            res = h[i];
          }
        }
      } else {
        res = std::max(res, findMaximum(H[trenutni], k));
      }
      std::swap(prosli, pretprosli);
      std::swap(trenutni, prosli);
    }
    
    cudaFree(A_device);
    cudaFree(B_device);
    delete [] h;
    for (int i = 0; i < 3; ++i) {
      cudaFree(&H[i]);
      cudaFree(&Ha[i]);
      cudaFree(&Hb[i]);
    }
  } catch (const Exception &ex) {
		ex.print();
	}

  return res;
}



/*!
 * \brief Smith-Waterman algorithm for bigger proteins. Can do it in
 *        reverse direction as well.
 */
void solveOnePhase(const nbrahle::Protein &first, const nbrahle::Protein &second, int block_size,
                   ResultType *results, bool silent=false) {
  int n = first.n();
  int m = second.n();
  SimpleMolecule *A;
  SimpleMolecule *B;
  ResultType *R;
  A = SimpleMoleculeToDevice(first, 0, n);
  B = SimpleMoleculeToDevice(second, 0, m);

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


/*!
 * \brief Does reconstruction of Smith-Waterman algorithm.
 */
void Reconstruct(ResultType *R, int x, int y, int n, int m,
                 const nbrahle::Protein &A, const nbrahle::Protein &B,
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



/*!
 * \brief Smith-Waterman algorithm, for bigger proteins.
 */
double smithWatermanCuda(nbrahle::Protein &first, nbrahle::Protein &second, bool silent=false, bool reconstruct=true) {
	cudaError_t cudaStatus;
  double res = 0;

  try {
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw CudaException(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		}

    int n = first.n();
    int m = second.n();

    ResultType *results = SimpleResultsInit(n, m);
    solveOnePhase(first, second, block_size, results, silent);
    int MX, MY;
    double mv=-1e100;
    FindMaxResult(mv, MX, MY, results, n, m);
    res = mv;

    if (reconstruct) { 
      nbrahle::Protein first_r(first);
      first_r.Resize(MX+1);
      first_r.Reverse();
      nbrahle::Protein second_r(second);
      second_r.Resize(MY+1);
      second_r.Reverse();

      ResultType *results2 = SimpleResultsInit(MX+1, MY+1);
      solveOnePhase(first_r, second_r, block_size, results2, silent);
      int mx, my;
      mv = -1e100;
      FindMaxResult(mv, mx, my, results2, MX+1, MY+1);
      res = mv;

      int top = MX-mx;
      int left = MY-my;
      int bottom = MX;
      int right = MY;
    
      if (!silent) {
        printf("Najbolje rjesenje mi je od (%d,%d) do (%d,%d)\n", top, left, bottom, right);
      }
      std::vector< std::pair< int, int > > solution;
      Reconstruct(results2, mx, my, MX+1, MY+1, first_r, second_r, solution);
      if (!silent) {
        OutputAllignment(solution);
      }
    }

    delete [] results;
  } catch (const Exception &ex) {
		ex.print();
	}

  return res;
}

