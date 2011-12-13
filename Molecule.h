#pragma once
#include <cstdio>
#include <string>
#include <iostream>

#include "cuda_runtime.h"

class Molecule
{
 public:
	Molecule();
	~Molecule();

  __device__ __host__ double x() const { return x_; }
  __device__ __host__ double y() const { return y_; }
  __device__ __host__ double z() const { return z_; }
  __device__ __host__ double deletion_cost() const { return deletion_cost_; }
  std::string name() const { return name_; }
  int position() const { return position_; }
  void set_position(int position) { position_ = position; }

  friend std::istream& operator>>(std::istream&, Molecule&);

 private:
  double x_, y_, z_;
	double deletion_cost_;
  std::string name_;
	int position_;
};

