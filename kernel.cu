
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>

#include "utilities.h"

#define INF 0x3f3f3f3f
#define CIJENA(x, i, j, k) (*((x) + (i)*m*3 + (j)*3 + (k)))
#define CIJENA2(x, i, j) (*((x) + (i)*m + (j)))
#define REZULTAT(x, i, j) (*((x) + (i)*m + (j)))

const int MAX_IME = 128;

cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);
cudaError_t smithWatermanCuda(int n, int m, char* prvi, char *drugi, int *rezultat, int *cijena);

struct Molekula {
	double x, y, z;
	double izbaci;
	char *ime;
	int m, pozicijaUProteinu;
	double *cijene;
	double *dev_cijene;

	Molekula() : x(0.0), y(0.0), z(0.0), izbaci(0), ime(0), m(0), pozicijaUProteinu(0), cijene(0) {
		ime = new char[MAX_IME];
	}

	double dajKvadratUdaljenosti(const Molekula &other) const {
		return (other.x - this->x) * (other.x - this->x) +
			   (other.y - this->y) * (other.y - this->y) +
			   (other.z - this->z) * (other.z - this->z);
	}

	void ucitaj(FILE *f, int pozicijaUProteinu=0) {
		fscanf(f, " %s%lf%lf%lf%lf", ime, &izbaci, &x, &y, &z);
		this->pozicijaUProteinu = pozicijaUProteinu;
	}

	void ucitajCijene(FILE *f, int m) {
		this->m = m;
		cijene = new double[m];
		for (int i = 0; i < m; ++i) {
			fscanf(f, "%lf", cijene+i);
		}
	}

	void alocirajCijeneNaCudi() {
		cudaError_t cudaStatus;
		if (!this->cijene) {
			return;
		}
		// prvo alociraj cijene na device
		size_t double_s = sizeof(double);
		cudaStatus = cudaMalloc((void**)&dev_cijene, double_s * this->m);
		if (cudaStatus != cudaSuccess) {
			throw CudaAllocationException(cudaStatus, "Nisam uspio alocirati niz cijena molekule");
		}
		// onda kopiraj na device
	    cudaStatus = cudaMemcpy(dev_cijene, cijene, double_s * m, cudaMemcpyHostToDevice);
	    if (cudaStatus != cudaSuccess) {
			throw CudaMemcpyException(cudaStatus, "Nisam uspio iskopoirati niz cijena");
		}
	}

	void cudaFree() {
		::cudaFree(dev_cijene);
	}
};

struct Protein {
	int n;
	Molekula *molekule;

	double dajKvadratUdaljenosti(int i, int j) const {
		return molekule[i].dajKvadratUdaljenosti(molekule[j]);
	}

	void ucitaj(FILE *f) {
		fscanf(f, "%d", &n);
		molekule = new Molekula[n];
		for (int i = 0; i < n; ++i) {
			molekule[i].ucitaj(f);
		}
	}

	Protein alocirajProteinNaCudi() {
		cudaError_t cudaStatus;

		// prvo svakoj molekuli prebaci cijene na device
		for (int i = 0; i < n; ++i) {
			molekule[i].alocirajCijeneNaCudi();
		}

		// onda alociraj niz molekula
		Molekula *dev_molekule;
		size_t molekula_s = sizeof(Molekula);
		cudaStatus = cudaMalloc((void**)&dev_molekule, molekula_s * n);
		if (cudaStatus != cudaSuccess) {
			throw CudaAllocationException("Nisam uspio alocirati niz molekula");
		}
		// onda kopiraj na device
	    cudaStatus = cudaMemcpy(dev_molekule, molekule, molekula_s * n, cudaMemcpyHostToDevice);
	    if (cudaStatus != cudaSuccess) {
			throw CudaMemcpyException("Nisam uspio iskopoirati niz molekula");
		}
		
		// na kraju napravi protein koji device moze koristiti
		Protein dev_protein;
		dev_protein.n = n;
		dev_protein.molekule = dev_molekule;
		return dev_protein;
	}

	void cudaFree() {
		for (int i = 0; i < n; ++i) {
			molekule[i].cudaFree();
		}
		::cudaFree(molekule);
	}
};

struct Info {
	int duzina, sirina;
	int dijagonala, dijagonalaOffset;

	Info() : duzina(0), sirina(0), dijagonala(0), dijagonalaOffset(0) {}
	Info(int _dijagonala, int _duzina=1, int _sirina=1, int _dijagonalaOffset=0) : duzina(_duzina), sirina(_sirina), dijagonala(_dijagonala), dijagonalaOffset(_dijagonalaOffset) {}
};


/*******************************************************
 * Funkcije koje se mogu koristiti za racunanje cijena *
 *******************************************************/
__device__ double dajKvadratUdaljenosti(const Molekula &A, const Molekula &B) {
	return (A.x-B.x)*(A.x-B.x) + (A.y-B.y)*(A.y-B.y) + (A.z-B.z)*(A.z-B.z);
}

__device__ double dajCijenuIzNiza(Molekula *A, Molekula *B) {
	return A->dev_cijene[B->pozicijaUProteinu];
}

/******************************************
 * Funkcija koja racuna tocno jedno polje *
 ******************************************/
#define FUNKCIJA dajCijenuIzNiza
__device__ void izracunaj(Protein prvi, Protein drugi, int i, int j, double *rezultat/*, double (*funkcija)(Molekula*, Molekula*)*/) {
	int n = prvi.n;
	int m = drugi.n;

	if (i > n || i < 0 || j > n || j < 0) return;
	Molekula *A = prvi.molekule + i;
	Molekula *B = drugi.molekule + j;

	double a = -INF, b = -INF, c = -INF;
	if (i == 0) {
		if (j == 0) {
			a = FUNKCIJA(A, B);
			b = A->izbaci;
			c = B->izbaci;
		} else {
			a = FUNKCIJA(A, B);
			b = A->izbaci;
			c = REZULTAT(rezultat, i, j-1) + B->izbaci;
		}
	} else if (j == 0) {
			a = FUNKCIJA(A, B);
			b = REZULTAT(rezultat, i-1, j) + A->izbaci;
			c = B->izbaci;
	} else {
			a = REZULTAT(rezultat, i-1, j-1) + FUNKCIJA(A, B);
			b = REZULTAT(rezultat, i-1, j) + A->izbaci;
			c = REZULTAT(rezultat, i, j-1) + B->izbaci;
	}

	if (a >= b && a >= c) {
		REZULTAT(rezultat, i, j) = a;
	}
	if (b >= a && b >= c) {
		REZULTAT(rezultat, i, j) = b;
	}
	if (c >= b && c >= a) {
		REZULTAT(rezultat, i, j) = c;
	}
}

/******************************************************
 * Kernel funkija koja racuna SW za pravokutnik polja *
 ******************************************************/
__global__ void swKernel2(Protein prvi, Protein drugi, Info info, double *rezultat) {
	int prvi_redak = (info.dijagonala - (threadIdx.x + info.dijagonalaOffset)) * info.duzina;
	int zadnji_redak = prvi_redak + info.duzina;
	int prvi_stupac = (threadIdx.x + info.dijagonalaOffset) * info.sirina;
	int zadnji_stupac = prvi_stupac + info.sirina;

	for (int i = prvi_redak; i < zadnji_redak; ++i) {
		for (int j = prvi_stupac; j < zadnji_stupac; ++j) {
			izracunaj(prvi, drugi, i, j, rezultat);
		}
	}
}










__global__ void swKernelFunc(Protein *prvi, Protein *drugi, int k, int *rezultat, int *cijena) {
	int i = k - threadIdx.x;
	int j = threadIdx.x;

	if (i > prvi->n || i < 0 || j > drugi->n || j < 0) return;
	Molekula *A = prvi->molekule + i;
	Molekula *B = drugi->molekule + j;
	int a = -INF, b = -INF, c = -INF;
	int n = prvi->n;
	int m = drugi->n;

	if (i == 0) {
		if (j == 0) {
			a = dajKvadratUdaljenosti(*A, *B);
			b = A->izbaci;
			c = B->izbaci;
		} else {
			a = dajKvadratUdaljenosti(*A, *B);
			b = A->izbaci;
			c = REZULTAT(rezultat, i, j-1) + B->izbaci;
		}
	} else if (j == 0) {
			a = dajKvadratUdaljenosti(*A, *B);
			b = REZULTAT(rezultat, i-1, j) + A->izbaci;
			c = B->izbaci;
	} else {
			a = REZULTAT(rezultat, i-1, j-1) + dajKvadratUdaljenosti(*A, *B);
			b = REZULTAT(rezultat, i-1, j) + A->izbaci;
			c = REZULTAT(rezultat, i, j-1) + B->izbaci;
	}

	if (a >= b && a >= c) {
		REZULTAT(rezultat, i, j) = a;
	}
	if (b >= a && b >= c) {
		REZULTAT(rezultat, i, j) = b;
	}
	if (c >= b && c >= a) {
		REZULTAT(rezultat, i, j) = c;
	}
}

__global__ void swKernel(char *prvi, char *drugi, int n, int m, int k, int *rezultat, const int *cijena) {
	int i = k - threadIdx.x;
	int j = threadIdx.x;

	if (i > n || i < 0 || j > m || j < 0) return;
	int a = -INF, b = -INF, c = -INF;
	if (i == 0) {
		if (j == 0) {
			a = CIJENA(cijena, i, j, 0);
			b = CIJENA(cijena, i, j, 1);
			c = CIJENA(cijena, i, j, 2);
		} else {
			a = CIJENA(cijena, i, j, 0);
			b = CIJENA(cijena, i, j, 1);
			c = REZULTAT(rezultat, i, j-1) + CIJENA(cijena, i, j, 2);
		}
	} else if (j == 0) {
			a = CIJENA(cijena, i, j, 0);
			b = REZULTAT(rezultat, i-1, j) + CIJENA(cijena, i, j, 1);
			c = CIJENA(cijena, i, j, 2);
	} else {
			a = REZULTAT(rezultat, i-1, j-1) + CIJENA(cijena, i, j, 0);
			b = REZULTAT(rezultat, i-1, j) + CIJENA(cijena, i, j, 1);
			c = REZULTAT(rezultat, i, j-1) + CIJENA(cijena, i, j, 2);
	}

	if (a >= b && a >= c) {
		REZULTAT(rezultat, i, j) = a;
	}
	if (b >= a && b >= c) {
		REZULTAT(rezultat, i, j) = b;
	}
	if (c >= b && c >= a) {
		REZULTAT(rezultat, i, j) = c;
	}
}

int main()
{
	FILE *f = fopen("input.txt", "r");
	int n, m;
	fscanf(f, "%d%d", &n, &m);
	char *prvi = new char[n+1];
	char *drugi = new char[m+1];
	fscanf(f, " %s %s", prvi, drugi);

	int *rezultat = new int[(n+1)*(m+1)];
	int *cijena = new int[(n+1)*(m+1)*3];

	memset(rezultat, 0, sizeof(int) * (n+1) * (m+1));

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			fscanf(f, "%d %d %d", &CIJENA(cijena, i, j, 0), &CIJENA(cijena, i, j, 1), &CIJENA(cijena, i, j, 2));
		}
	}
	
	cudaError_t cudaStatus = smithWatermanCuda(n, m, prvi, drugi, rezultat, cijena);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "smithWatermanCuda failed!");
		system("pause");
        return 1;
    }

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			printf("%5d ", REZULTAT(rezultat, i, j));
		}
		printf("\n");
	}

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

void smithWatermanCuda3(Protein prvi, Protein drugi, double *rezultat, double *cijena) {
	cudaError_t cudaStatus;
	size_t double_s = sizeof(double);
	double *dev_rezultat;

	int n = prvi.n;
	int m = drugi.n;

	try {
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw Exception("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		}

		// Alociraj prvi i drugi protein na cudi.
		Protein dev_prvi = prvi.alocirajProteinNaCudi();
		Protein dev_drugi = drugi.alocirajProteinNaCudi();

		// Alociraj i rezultat na cudi.
		dev_rezultat = kopirajArrayNaDevice(rezultat, (n+1)*(m+1));

		// vrti petlju
		for (int i = 0; i < n+m+1; ++i) {
			Info info = Info(i);
			swKernel2<<< 1, i+1 >>>(dev_prvi, dev_drugi, info, dev_rezultat);
			sinkronizirajCudaDretve();
		}

		// Vrati rezultat natrag na host.
		kopirajArrayNaHost(rezultat, dev_rezultat, (n+1)*(m+1));
	} catch (const Exception &ex) {
		ex.print();
	}

	prvi.cudaFree();
	drugi.cudaFree();
	cudaFree(dev_rezultat);
}

void smithWatermanCuda2(int n, int m, Protein prvi, Protein drugi, double *rezultat) {
	cudaError_t cudaStatus;
	size_t double_s = sizeof(double);
	double *dev_rezultat;

	try {
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw Exception("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		}

		// Alociraj prvi i drugi protein na cudi.
		Protein dev_prvi = prvi.alocirajProteinNaCudi();
		Protein dev_drugi = drugi.alocirajProteinNaCudi();

		// Alociraj i rezultat na cudi.
		cudaStatus = cudaMalloc((void**)&dev_rezultat, double_s * (n+1) * (m+1));
		if (cudaStatus != cudaSuccess) {
			throw CudaAllocationException("nisam uspio alocirati polje za rezultate");
		}
	    cudaStatus = cudaMemcpy(dev_rezultat, rezultat, double_s * (n+1) * (m+1), cudaMemcpyHostToDevice);
	    if (cudaStatus != cudaSuccess) {
		    throw CudaMemcpyException("nisam uspio iskopirati poolje s rezultatima");
		}

		for (int i = 0; i < n+m+1; ++i) {
			Info info = Info(i);
			swKernel2<<< 1, i+1 >>>(dev_prvi, dev_drugi, info, dev_rezultat);
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				throw Exception("cudaDeviceSynchronize je vratila pogresku nakon lansiranja kernela");
			}
		}

		// Vrati rezultat natrag na host.
		cudaStatus = cudaMemcpy(rezultat, dev_rezultat, double_s * (n+1) * (m+1), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw Exception("Nisam uspio vratiti rezultat na domacina");
		}

	} catch (const Exception &ex) {
		ex.print();
	}

	prvi.cudaFree();
	drugi.cudaFree();
	cudaFree(dev_rezultat);
}

cudaError_t smithWatermanCuda(int n, int m, char* prvi, char *drugi, int *rezultat, int *cijena) {
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	// Allocate the memory.
	size_t char_s = sizeof(char);
	char *dev_prvi;
	char *dev_drugi;
	
	cudaStatus = cudaMalloc((void**)&dev_prvi, char_s * (n+1));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on allocating first string!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_drugi, char_s * (n+1));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on allocating second string!");
        goto Error;
    }

	size_t int_s = sizeof(int);
	int *dev_cijena;
	int *dev_rezultat;

	cudaStatus = cudaMalloc((void**)&dev_cijena, int_s * (n+1) * (m+1) * 3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on allocating cost array!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_rezultat, int_s * (n+1) * (m+1));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on allocating results array!");
        goto Error;
    }


	// Time to copy the memory.
    cudaStatus = cudaMemcpy(dev_prvi, prvi, char_s * (n+1), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on first string!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_drugi, drugi, char_s * (m+1), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on second string!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_cijena, cijena, int_s * (n+1) * (m+1) * 3, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on cost array!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_rezultat, rezultat, int_s * (n+1) * (m+1), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on result array!");
        goto Error;
    }

	for (int i = 0; i < n+m+1; ++i) {
		printf("%d\n", i);
		swKernel<<< 1, i+1 >>>(dev_prvi, dev_drugi, n, m, i, dev_rezultat, dev_cijena);
	    //addKernel<<< 1, size >>>(dev_c, dev_a, dev_b);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching swKernel!\n", cudaStatus);
			goto Error;
		}
	}

	// Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(rezultat, dev_rezultat, int_s * (n+1) * (m+1), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


Error:
	cudaFree(dev_prvi);
	cudaFree(dev_drugi);
	cudaFree(dev_cijena);
	cudaFree(dev_rezultat);

	return cudaStatus;
}

