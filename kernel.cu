// TODO: uredi ove includove

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>

#include "utilities.h"
#include "typedefs.h"
#include "Protein.h"
#include "Molecule.h"
#include "Info.h"
#include "Results.h"


void smithWatermanCuda(Protein, Protein);


__device__ double DistCost(const Molecule &A, const Molecule &B) {
  return sqr(A.x() - B.x()) + sqr(A.y() - B.y()) + sqr(A.z() - B.z());
}


__global__ void OneElement(Protein protein_A, Protein protein_B, int k,
                           Results R) {
  int i = k - threadIdx.x;
  int j = threadIdx.x;

  const Molecule &molecule_A = protein_A[i];
  const Molecule &molecule_B = protein_B[i];

  ResultsType a, b, c;
  a = R.GetResult(i-1, j-1); // TODO: ovdje fali cijena
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


int main()
{
  // TODO: ovdje nesto fali
  cudaError_t cudaStatus;

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


void smithWatermanCuda(Protein prvi, Protein drugi) {
	cudaError_t cudaStatus;
  Results results;
	Results dev_results;

	int n = prvi.n();
	int m = drugi.n();
  results.Init(n, m);

  try {
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw CudaException(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		}

		// Alociraj prvi i drugi protein na cudi.
		Protein dev_prvi = prvi.createCopyOnCuda();
		Protein dev_drugi = drugi.createCopyOnCuda();

		// Alociraj i rezultat na cudi.
    dev_results = results.CreateCopyOnDevice();

		// vrti petlju
		for (int i = 0; i < n+m+1; ++i) {
			OneElement<<< 1, i+1 >>>(dev_prvi, dev_drugi, i, dev_results);
			syncCudaThreads();
		}

		// Vrati rezultat natrag na host.
    results = dev_results.CreateCopyOnHost();
	} catch (const Exception &ex) {
		ex.print();
	}
}
