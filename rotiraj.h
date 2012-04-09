#pragma once
#include <cstdio>
#include <cmath>

const double PI = 3.1415926535897932384626433832795;

struct Point3D {
  double C[3];
  Point3D() {}
  Point3D(double x, double y, double z) {
    C[0] = x;
    C[1] = y;
    C[2] = z;
  }
};

Point3D operator+(Point3D A, const Point3D &B);
struct RotationMatrix {
  double P[3][3];
};

Point3D operator*(const RotationMatrix &A, const Point3D &B);

RotationMatrix operator*(const RotationMatrix &A, const RotationMatrix &B);

RotationMatrix rotateX(double theta);

RotationMatrix rotateY(double theta);

RotationMatrix rotateZ(double theta);

RotationMatrix createRotationMatrix(double thetaX, double thetaY, double thetaZ);

Point3D dajPomak(double x, double y, double z);
