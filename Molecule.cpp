#include "Molecule.h"

#include <iostream>
#include <string>

Molecule::Molecule()
    : x_(0),
      y_(0),
      z_(0),
      deletion_cost_(0),
      name_(),
      position_(0)
{
}


Molecule::~Molecule()
{
}


std::istream& operator>>(std::istream& in, Molecule& molekula) {
  in >> molekula.name_;
  in >> molekula.deletion_cost_;
  in >> molekula.x_ >> molekula.y_ >> molekula.z_;
  return in;
}

/*
// Implemented in the kernel.cu file
__device__ __host__ double Molecule::x() const { return x_; }
__device__ __host__ double Molecule::y() const { return y_; }
__device__ __host__ double Molecule::z() const { return z_; }
__device__ __host__ double Molecule::deletion_cost() const { return deletion_cost_; }
*/