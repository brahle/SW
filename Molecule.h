#pragma once
#include <cstdio>
#include <string>
#include <iostream>

#include "cuda_runtime.h"

class Molecule
{
 public:
	Molecule();
  Molecule(const Molecule&);
	~Molecule();

  double x() const;
  double y() const;
  double z() const;
  double deletion_cost() const;
  int name() const { return name_; }
  int position() const { return position_; }
  void set_position(int position) { position_ = position; }
  void set_x(double x) { x_ = x; }
  void set_y(double y) { y_ = y; }
  void set_z(double z) { z_ = z; }

  friend std::istream& operator>>(std::istream&, Molecule&);

 private:
  double x_, y_, z_;
	double deletion_cost_;
  int name_;
	int position_;
};

