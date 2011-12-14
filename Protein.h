#pragma once
#include <iostream>

#include "cuda_runtime.h"

#include "Molecule.h"
#include "utilities.h"

class Protein
{
 public:
	Protein(void);
  Protein(int, Molecule*, bool);
	~Protein(void);

  Protein createCopyOnCuda() const;

  int n() const { return n_; }
  Molecule* molekule() const { return molecules_; }
  __device__ __host__ Molecule& operator[](int i);

  friend std::istream& operator>>(std::istream&, Protein);

 private:
  int n_;
  Molecule* molecules_;
  bool on_cuda_;
};

