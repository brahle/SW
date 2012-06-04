#pragma once
#include <algorithm>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"

#include "Molecule.h"
#include "utilities.h"
#include "rotiraj.h"

namespace nbrahle { 

class Protein
{
 public:
	Protein(void);
  Protein(int, Molecule*, bool);
  Protein(const Protein&);
  Protein(const std::vector< Point3D >&);
	~Protein(void);

  Protein createCopyOnCuda() const;

  __device__ __host__ int n() const;
  int end() const { return end_; }
  void SetEnd(int end) { end_ = end; }
  void Reverse();
  void Resize(int newSize);
  Molecule* molekule() const { return molecules_; }
  __device__ __host__ Molecule& operator[] (int i);
  __device__ __host__ Molecule& operator[] (int i) const;

  double* CopyXToDevice() const;
  double* CopyYToDevice() const;
  double* CopyZToDevice() const;
  double* CopyDCToDevice() const;

  friend std::istream& operator>>(std::istream&, nbrahle::Protein&);
  friend std::ostream& operator<<(std::ostream&, nbrahle::Protein&);

 private:
  int n_;
  Molecule* molecules_;
  bool on_cuda_;
  int end_;
};
}
