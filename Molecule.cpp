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
