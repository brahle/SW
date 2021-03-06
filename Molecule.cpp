#include "Molecule.h"

#include <iostream>
#include <string>
#include "rotate.h"

using namespace nbrahle;

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


Molecule::Molecule(const Point3D &P, int name, int position)
    : x_(P.C[0]),
      y_(P.C[1]),
      z_(P.C[2]),
      deletion_cost_(0),
      name_(name),
      position_(position)
{
}


Molecule::~Molecule()
{
}


std::istream& ::operator>>(std::istream& in, Molecule& molekula) {
  in >> molekula.name_;
  in >> molekula.deletion_cost_;
  in >> molekula.x_ >> molekula.y_ >> molekula.z_;
  return in;
}

std::ostream& ::operator<<(std::ostream& out, Molecule& molekula) {
  out << "(" << molekula.x_ << "," << molekula.y_ << "," << molekula.z_ << ")";
  return out;
}

/*
// Implemented in the utilities.cuh file
__device__ __host__ double Molecule::x() const { return x_; }
__device__ __host__ double Molecule::y() const { return y_; }
__device__ __host__ double Molecule::z() const { return z_; }
__device__ __host__ double Molecule::deletion_cost() const { return deletion_cost_; }
*/