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


Protein::~Protein(void)
{
  if (on_cuda_) {
    cudaFree(molecules_);
  } else {
    delete [] molecules_;
  }
}


Protein Protein::createCopyOnCuda() const {
  return Protein(n_, copyArrayToDevice(molecules_, n_), true);
}


std::istream& operator>>(std::istream& in, Protein p) {
  in >> p.n_;
  p.molecules_ = new Molecule[p.n_];
  for (int i = 0; i < p.n_; ++i) {
    in >> p.molecules_[i];
  }
  return in;
}
