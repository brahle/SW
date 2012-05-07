#include "Protein.h"

#include "utilities.h"
#include "cuda_runtime.h"

Protein::Protein(void)
  : n_(0),
    molecules_(0),
    on_cuda_(false)
{
}


Protein::Protein(int n, Molecule* molecules, bool on_cuda=false)
  : n_(n),
    molecules_(molecules),
    on_cuda_(on_cuda)
{
}


Protein::Protein(const Protein &P)
  : n_(P.n()),
    on_cuda_(P.on_cuda_)
{
  molecules_ = new Molecule[n_];
  memcpy(molecules_, P.molecules_, sizeof(Molecule) * n_);
}


Protein::Protein(const std::vector< Point3D > &P)
  : n_(P.size()),
    on_cuda_(false)
{
  molecules_ = new Molecule[n_];
  for (int i = 0; i < n_; ++i) {
    molecules_[i] = Molecule(P[i], i, i);
  }
}


Protein::~Protein(void)
{
  if (on_cuda_) {
    cudaFree(molecules_);
  } else {
    delete [] molecules_;
  }
}


void Protein::Resize(int newSize) {
  Molecule *new_molecules = new Molecule[newSize];
  memcpy(new_molecules, molecules_, std::min(newSize, n_) * sizeof(Molecule));
  delete [] molecules_;
  molecules_ = new_molecules;
  n_ = newSize;
}

void Protein::Reverse() {
  for (int i = 0; i*2 < n_; ++i) {
    std::swap(molecules_[i], molecules_[n_-i-1]);
  }
}

/*
Protein Protein::createCopyOnCuda() const {
  return Protein(n_, copyArrayToDevice(molecules_, n_), true);
}
*/

/*
// Implemented in the kernel.cu file
__device__ __host__ Molecule& Protein::operator[](int i) { return molecules_[i]; }
*/


std::istream& operator>>(std::istream& in, Protein& p) {
  in >> p.n_;
  p.molecules_ = new Molecule[p.n_];
  for (int i = 0; i < p.n_; ++i) {
    in >> p.molecules_[i];
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, Protein& p) {
  out << p.n_ << ": ";
  for (int i = 0; i < p.n_; ++i) {
    if (i) out << ", ";
    out << p.molecules_[i];
  }
  return out;
}

