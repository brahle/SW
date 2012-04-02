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

Point3D operator+(Point3D A, const Point3D &B) {
  static int n = 3;
  for (int i = 0; i < n; ++i)
    A.C[i] += B.C[i];
  return A;
}

struct RotationMatrix {
  double P[3][3];

};

Point3D operator*(const RotationMatrix &A, const Point3D &B) {
  Point3D ret;
  static int n = 3;
  for (int i = 0; i < n; ++i) {
    ret.C[i] = 0.0;
    for (int j = 0; j < n; ++j) {
      ret.C[i] += A.P[i][j] * B.C[j];
    }
  }
  return ret;
}

RotationMatrix operator*(const RotationMatrix &A, const RotationMatrix &B) {
  RotationMatrix ret;
  static int n = 3;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      ret.P[i][j] = 0.0;
      for (int k = 0; k < n; ++k) {
        ret.P[i][j] += A.P[i][k] * B.P[k][j];
      }
    }
  }
  return ret;
}

RotationMatrix rotateX(double theta) {
  RotationMatrix ret;
  ret.P[0][0] = 1.0;         ret.P[0][1] = 0.0;         ret.P[0][2] = 0.0;
  ret.P[1][0] = 0.0;         ret.P[1][1] = cos(theta);  ret.P[1][2] = -sin(theta);
  ret.P[2][0] = 0.0;         ret.P[2][1] = sin(theta);  ret.P[2][2] = cos(theta);
  return ret;
}

RotationMatrix rotateY(double theta) {
  RotationMatrix ret;
  ret.P[0][0] = cos(theta);  ret.P[0][1] = 0.0;         ret.P[0][2] = sin(theta);
  ret.P[1][0] = 0.0;         ret.P[1][1] = 1.0;         ret.P[1][2] = 0.0;
  ret.P[2][0] = -sin(theta); ret.P[2][1] = 0.0;         ret.P[2][2] = cos(theta);
  return ret;
}

RotationMatrix rotateZ(double theta) {
  RotationMatrix ret;
  ret.P[0][0] = cos(theta);  ret.P[0][1] = -sin(theta); ret.P[0][2] = 0.0;
  ret.P[1][0] = sin(theta);  ret.P[1][1] = cos(theta);  ret.P[1][2] = 0.0;
  ret.P[2][0] = 0.0;         ret.P[2][1] = 0.0;         ret.P[2][2] = 1.0;
  return ret;
}

RotationMatrix createRotationMatrix(double thetaX, double thetaY, double thetaZ) {
  return (rotateX(thetaX) * rotateY(thetaY)) * rotateZ(thetaZ);
}

Point3D dajPomak(double x, double y, double z) {
  return Point3D(x, y, z);
}

int main() {
  Point3D A;
  double alfa, beta, gama;
  double dx, dy, dz;

  scanf("%lf%lf%lf", A.C, A.C+1, A.C+2);
  scanf("%lf%lf%lf", &alfa, &beta, &gama);
  scanf("%lf%lf%lf", &dx, &dy, &dz);
  alfa = alfa / 180.0 * PI;
  beta = beta / 180.0 * PI;
  gama = gama / 180.0 * PI;

  RotationMatrix rotacija = createRotationMatrix(alfa, beta, gama);
  Point3D pomak = dajPomak(dx, dy, dz);
  Point3D B = rotacija * A + pomak;
  printf("%g %g %g\n", B.C[0], B.C[1], B.C[2]);

  return 0;
}

