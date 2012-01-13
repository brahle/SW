#include "Molecule.h"

#include <iostream>
#include <string>

Molecule::Molecule()
    : x_(0),
      y_(0),
      z_(0),
      deletion_cost_(0),
      name_(0),
      position_(0)
{
}


Molecule::Molecule(const Molecule &M)
    : x_(M.x_),
      y_(M.y_),
      z_(M.z_),
      deletion_cost_(M.deletion_cost_),
      name_(M.name_),
      position_(M.position_)
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