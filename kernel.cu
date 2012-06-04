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

#include "utilities.h"
#include "typedefs.h"
#include "Protein.h"
#include "Molecule.h"
#include "rotiraj.h"
#include "consts.h"

using namespace nbrahle;

typedef struct {
    char type [PDB_ATOM_ATOM_NAME_LEN+1];
    double x,y,z;
    int backbone;
} Atom;

typedef struct {
    char pdb_id[PDB_ATOM_RES_NO_LEN+2];
    char res_type[PDB_ATOM_RES_NAME_LEN+1];
    char res_type_short;
    char chain;
    int no_atoms;
    Atom atom[MAX_NO_ATOMS];
    Atom *Ca;
    int interface;
    int solvent_accessible;
    int belongs_to_helix;
    int belongs_to_strand;
    int alt_belongs_to_helix; /* we allow a couple of residues overlap in SSEs */
    int alt_belongs_to_strand;
} Residue;

typedef struct {} SSElement;

typedef struct {
    int length;
    Residue * sequence;
    int no_helices;
    SSElement *helix;
    int no_strands;
    SSElement *strand;
    int * sse_sequence;
} Protein;

typedef struct {
    ////////////////////
    // general
    int size;                // max number of mapped elements (SSEs or Ca, depending where we use the structure
    int matches;             // the number ofactually mapped SSEs
    double q[4]; /* rotation (quaternion)  -- the rotation which results in this particular map */
    double T[3]; /* translation (once we believe we know it)  -- the translation (for the Ca level)*/
    ////////////////////
    // referring to SSE-level map:
    int *x2y, *y2x;          // x2y: for each mapped pair (x,y), returns y, the index of the SSE in the structure Y,
                             // given x, the index of SSE in structure X  ( x2y[x] = y, e.g. x2y[3] = 7)
    int x2y_size, y2x_size;  // the size of the two maps above; ultimately the two should be equal
                             // in some general intermediate step that might not be the case, in the
                             // current implementation it always is
    double avg_length_mismatch; // average difference in the length of mapped SSEs
    double rmsd;             /* rmsd for the centers of matched SSEs */
    ////////////////////
    // "urchin" scoring
    double **cosine;          // table of angle cosines for all pairs (x,y) (all of them, not just the mapped ones)
    double **image;           // table of exp terms for all pairs (x,y) (all of them, not just the mapped ones)
    double F;                 // value of the  function F for this map
    double avg, avg_sq;       // refers to the avg and average square of F over the sapce of all rotations
                              // the input for the calulcation of the z-score
    double z_score;           // z-score for F (based on avg and avg_sq0
    double assigned_score;    // sum of the exp terms of but only for the  matched SSE pairs (x,y)
    ////////////////////
    // referring to Ca-level map:
    int *x2y_residue_level, *y2x_residue_level;  // the same as the x2y above, this time not on SSE, but on Ca level
    int x2y_residue_l_size, y2x_residue_l_size;
    int res_almt_length;    // length of the alignment on the Ca level
    double res_rmsd;        /* rmsd for the matched Ca atoms*/
    double aln_score;         // like the "assigned score" above, but for the Ca level
    double res_almt_score;/*not sure - cou;ld be I am duplicating aln_score */
    ////////////////////

    // complementary or sub-maps - never mind for now, just leave as is
    int *submatch_best;         // the best map which complements this one; don't worry about it right now
    double score_with_children; // this goes with the submatch above = never mind
    double compl_z_score;       // z-score for the submatch
    ///////////////////

    // file to which the corresponding pdb was written
    char filename[MEDSTRING];
} Map;


struct SimpleMolecule { double x, y, z, dc; };
double smithWatermanCuda(nbrahle::Protein&, nbrahle::Protein&, bool, bool);
double smithWatermanCudaFast(nbrahle::Protein&, nbrahle::Protein&);
SimpleMolecule* SimpleMoleculeToDevice(const nbrahle::Protein &, int, int, bool);

struct Stanje {
  std::vector< Point3D > original, P;
  double dx, dy, dz;
  double thetaX, thetaY, thetaZ;

  Stanje() : original(), P(), dx(0.0), dy(0.0), dz(0.0), thetaX(0.0), thetaY(0.0), thetaZ(0.0) {}

  Stanje(::Protein *p) : original(), P(), dx(0.0), dy(0.0), dz(0.0), thetaX(0.0), thetaY(0.0), thetaZ(0.0) {
    // TODO: implement this
  }

  void ucitaj(const char *file_name) {
    FILE *f = fopen(file_name, "r");
    char buff[1024];

    printf("Ucitavam podatke iz %s...\n", file_name);
    while (fgets(buff, sizeof(buff), f)) {
      double x, y, z;
      if (sscanf(buff, "ATOM %*d CA %*s %*s %*d %lf %lf %lf", &x, &y, &z)==3) {
        P.push_back(Point3D(x,y,z));
      }
    }
    original = P;
  }

  void tresni(double maxTranslacija=5.0, double maxRotacija=0.1) {
    dx = maxTranslacija - fmod(rand()*1.0, 2*maxTranslacija);
    dy = maxTranslacija - fmod(rand()*1.0, 2*maxTranslacija);
    dz = maxTranslacija - fmod(rand()*1.0, 2*maxTranslacija);
    thetaX = maxRotacija - 2*maxRotacija*rand()/RAND_MAX;
    thetaY = maxRotacija - 2*maxRotacija*rand()/RAND_MAX;
    thetaZ = maxRotacija - 2*maxRotacija*rand()/RAND_MAX;
    RotationMatrix rotacija = createRotationMatrix(thetaX, thetaY, thetaZ);
    Point3D pomak = dajPomak(dx, dy, dz);

    P.clear();
    for (int i = 0; i < (int)original.size(); ++i) {
      Point3D pomaknuta_tocka = rotacija * original[i] + pomak;
      P.push_back(pomaknuta_tocka);
    }
  }

  void pomakni(const Stanje &A) {
    dx = A.dx + 5 - rand() % 11;
    dy = A.dy + 5 - rand() % 11;
    dz = A.dz + 5 - rand() % 11;
    thetaX = A.thetaX + 0.1 - 0.2*rand()/RAND_MAX;
    thetaY = A.thetaY + 0.1 - 0.2*rand()/RAND_MAX;
    thetaZ = A.thetaZ + 0.1 - 0.2*rand()/RAND_MAX;
    RotationMatrix rotacija = createRotationMatrix(thetaX, thetaY, thetaZ);
    Point3D pomak = dajPomak(dx, dy, dz);

    original = A.original;
    P.clear();
    for (int i = 0; i < (int)A.original.size(); ++i) {
      Point3D pomaknuta_tocka = rotacija*A.original[i] + pomak;
      P.push_back(pomaknuta_tocka);
    }
  }

  void reset() {
    dx = 0;
    dy = 0;
    dz = 0;
    thetaX = 0;
    thetaY = 0;
    thetaZ = 0;
    RotationMatrix rotacija = createRotationMatrix(thetaX, thetaY, thetaZ);
    Point3D pomak = dajPomak(dx, dy, dz);
    P.clear();
    for (int i = 0; i < (int)original.size(); ++i) {
      Point3D pomaknuta_tocka = rotacija * original[i] + pomak;
      P.push_back(pomaknuta_tocka);
    }
  }
};

Stanje getNeighbour(const Stanje &A) {
  Stanje ret;
  ret.pomakni(A);
  return ret;
}

Stanje original;

double getEnergy(const Stanje &other) {
  nbrahle::Protein p1(original.P), p2(other.P);
  return -smithWatermanCudaFast(p1, p2);
}

template< typename StateType, typename EnergyFunction, typename NextFunction >
StateType annealing(const StateType &original, EnergyFunction getEnergy,
                    NextFunction neighbour, int maxEvolution,
                    double acceptable_energy, int pool_size=1) {
  StateType current_state = original, *next_states = new StateType[pool_size];
  double current_energy = getEnergy(current_state), *next_energies = new double[pool_size];
  int next_best;

  StateType best_state = current_state;
  double best_energy = current_energy;

  int reset_temperature = 0;
  int promjena_pred = 0;

  for (int i = 0; i < maxEvolution; ++i) {
    if (best_energy <= acceptable_energy) break;
    double T = temperature(i-reset_temperature);

    next_best = 0;
    if (rand() < 0.1 * RAND_MAX) {
      next_states[0] = neighbour(current_state);
      next_energies[0] = getEnergy(next_states[0]);
    } else {
      for (int j = 0; j < pool_size; ++j) {
        next_states[j] = neighbour(current_state);
        next_energies[j] = getEnergy(next_states[j]);
        if (next_energies[j] < next_energies[next_best]) {
          next_best = j;
        }
      }
    }
    printf("%d. Best = %g; Curr = %g; Neigh = %g; Temp = %g\n", i+1, best_energy, current_energy, next_energies[next_best], T);

    double a = P(current_energy, next_energies[next_best], T);
    double b = rand();
    //printf("S %g na %g uz temperaturu %g: vjerojatnost je %g\n", current_energy, next_energies[next_best], T, a*100.0);
    if (a*RAND_MAX > b) {
      current_state = next_states[next_best];
      current_energy = next_energies[next_best];
      promjena_pred = 0;
    } else {
      ++promjena_pred;
      if (promjena_pred == 15) {
        reset_temperature = i+1;
        promjena_pred = 0;
      }
    }
    if (current_energy < best_energy) {
      best_state = current_state;
      best_energy = current_energy;
    }

  }

  printf("Konacna najbolja energija = %g\n", best_energy);
  delete [] next_states;
  delete [] next_energies;
  return best_state;
}

void testMinimum(const int);



int main() {
  // Ucitaj podatke
  cudaError_t cudaStatus;
  Stanje prvi, drugi;
    
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		throw CudaException(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

//  prvi.ucitaj("1d0nA.pdb");
//  drugi.ucitaj("2d8bA.pdb");

  prvi.ucitaj("1a0iA.pdb");
  printf("Rotiram ulazni niz...\n");
  drugi = prvi;
  drugi.tresni(10, 1.5);


  original = prvi;
  
  printf("Najbolja teoretska energija: %g\n", getEnergy(original));

  double start = clock();
  // Izracunaj rezultat
  Stanje result = annealing(drugi, getEnergy, getNeighbour, STEPS, -1e100, POOL_SIZE);
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





int single_map_optimize_bb_almt (::Protein *p1, ::Protein *p2, Map * map) {
  /*
    the current best guess for the rotation and translation are in
    map->q (the rotation representeed as a 4-component quaternion; the components defined as doubles),
    and  map->T (3 component; double); to get the rotation matrix use 
    quat_to_R (q, R); defined in  04_geometric_match/struct_quaternion.c:30
    Map is defined in 00_include/struct.h:190      
  */

  Stanje S1(p1), S2(p2);
  original = S1;
  Stanje res = annealing(S2, getEnergy, getNeighbour, 100, -1e100, 20);
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











void SyncCudaThreads() {
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		throw CudaException(cudaStatus, "cudaDeviceSynchronize je vratila pogresku nakon lansiranja kernela");
	}
}


template <typename T> __device__ T sqr(const T& A) {
  return A*A;
}

__device__  ResultType GetResult(ResultType *R, int n, int m, int i, int j) {
  if (i < -1 || j < -1 || i >= n || j >= m) return ResultType(-1.0, 0);
  return R[(i+1)*(m+1) + j+1];
}

__device__ void SetResult(ResultType *R, int n, int m, int i, int j,
                          ResultType val) {
  if (i < 0 || j < 0 || i >= n || j >= m) return;
  R[(i+1)*(m+1) + j+1] = val;
}

inline __device__ double DistCost(const SimpleMolecule &A, const SimpleMolecule &B) {
  return 1000 - exp((sqr(A.x - B.x) + sqr(A.y - B.y) + sqr(A.z - B.z))/100);
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

template< typename T>
inline __device__ T min(const T &a, const T &b) { return a < b ? a : b; }

template< typename T>
inline __device__ T max(const T &a, const T &b) { return a > b ? a : b; }

inline __device__ int getIndex(int i, int j, int n, int m) {
  if (i < 0 || j < 0) return n+m;
  int d = i+j;
  if (d < m) return i;
  return i+m-d-1;
}








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

template< typename T >
__global__ void reduceMax(const T *g_idata, T *g_odata) {
  extern __shared__ T sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  for (unsigned int s=1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] = max(sdata[index], sdata[index + s]);
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void dumbReduceMax(const double *data, double *res, int n) {
  *res = data[0];
  for (int i = 1; i < n; ++i) {
    if (*res < data[i]) {
      *res = data[i];
    }
  }
}

double findMaximumDumb(const double *device_data, const int n) {
  double *izlaz_d, *izlaz_h;
  double res = 0;
  cudaError_t cudaStatus = cudaMalloc(&izlaz_d, sizeof(double));

  dumbReduceMax<<< 1, 1 >>>(device_data, izlaz_d, n);
  izlaz_h = copyArrayToHost(izlaz_d, 1);

  cudaFree(izlaz_d);
  res = izlaz_h[0];
  delete [] izlaz_h;
  return res;
}

double findMaximum(double *device_data, const int n) {
  double *izlaz_d;
  double res = 0;
  cudaError_t cudaStatus = cudaMalloc(&izlaz_d, n*sizeof(double));

  if (n < 4192) {
    return findMaximumDumb(device_data, n);
  }

  int m = 0, cnt = 0;
  for (int sz = 1; sz <= n; sz <<= 1) {
    if (sz & n) {
      int blocks = std::max(1, sz/512);
      int threads = std::min(sz, 512);
      int shared_mem = threads*sizeof(double);
      reduceMax<<< blocks, threads, shared_mem >>>(device_data+cnt, izlaz_d+m);
      cnt += sz;
      m += blocks;
    }
  }
  
  res = findMaximumDumb(izlaz_d, m);
  cudaFree(izlaz_d);
  return res;
}

void testMaximum(const int n) {
  double *data_host = new double[n], *data_device;
  srand(time(0));
  int id = 0;

  for (int i = 0; i < n; ++i) {
    data_host[i] = rand();
    if (data_host[id] < data_host[i]) {
      id = i;
    }
  }
  
  data_device = copyArrayToDevice(data_host, n);
  double r = findMaximum(data_device, n);
  assert(abs(data_host[id]-r) < 0.5);
  delete [] data_host;
  cudaFree(data_device);
}

int brojElemenataUDijagonali(int n, int m, int d) {
  return std::min(std::min(n, m), std::min(d+1, n+m-1-d));
}

int maxElemenataNaDijagonali(int n, int m) {
  return std::min(n, m);
}

double smithWatermanCudaFast(nbrahle::Protein &first, nbrahle::Protein &second) {
	cudaError_t cudaStatus;
  double res = -1e100;

  try {
		// Choose which GPU to run on, change this on a multi-GPU system.

    int n = first.n();
    int m = second.n();
    int l = maxElemenataNaDijagonali(n, m)+1;
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
      int k = brojElemenataUDijagonali(n, m, d);
      int blocks = (k+511)/512;
      int threads = std::min(k, 512);

      StepFast<<< blocks, threads >>>( H[trenutni],   Ha[trenutni],   Hb[trenutni],
                                       H[prosli],     Ha[prosli],     Hb[prosli],
                                       H[pretprosli], Ha[pretprosli], Hb[pretprosli],
                                       A_device, B_device, n, m, d, k,
                                       PROCJEP_POCETAK, PROCJEP_NASTAVAK);
      SyncCudaThreads();
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














void PrintResults(ResultType *R, int n, int m) {
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= m; ++j) {
      printf("%9g", R[i*(m+1) + j].value);
    }
    printf("\n");
  }
}


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

void Output(const std::vector< std::pair< int, int > > &A) {
  for (int i = 0; i < (int)A.size(); ++i) {
    if (A[i].first == -1) printf("%8s", "-");
    else printf("%8d", A[i].first);
    if (A[i].second == -1) printf("%8s", "-");
    else printf("%8d", A[i].second);
    printf("\n");
  }
}

void parallelFindMax(double &max_value, int &mx, int &my, ResultType *results, int n, int m) {
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
    parallelFindMax(mv, MX, MY, results, n, m);
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
      parallelFindMax(mv, mx, my, results2, MX+1, MY+1);
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


nbrahle::Protein nbrahle::Protein::createCopyOnCuda() const {
  return nbrahle::Protein(n_, copyArrayToDevice(molecules_, n_), true);
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

__device__ __host__ Molecule& nbrahle::Protein::operator[](int i) { return molecules_[i]; }
__device__ __host__ Molecule& nbrahle::Protein::operator[](int i) const { return molecules_[i]; }

__device__ __host__ RESULTTYPE::RESULTTYPE() : value(0), move(-1) {}
__device__ __host__ RESULTTYPE::RESULTTYPE(ResultValue v, int m): value(v), move(m) {}

__device__ __host__ int nbrahle::Protein::n() const { return n_; }


template <typename T> void allocArrayOnDevice(T **ptr, int size) {
	cudaError_t cudaStatus = cudaMalloc((void**)ptr, sizeof(T)*size);
	if (cudaStatus != cudaSuccess) {
		throw Exception("nisam uspio alocirati polje na uredjaju");
	}
}

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

