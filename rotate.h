#pragma once
#include <cstdio>
#include <cmath>

const double PI = 3.1415926535897932384626433832795;

/*!
 * \brief Raw data structe used to hold a point in 3D space.
 */
struct Point3D {
  double C[3];
  Point3D() {}
  Point3D(double x, double y, double z) {
    C[0] = x;
    C[1] = y;
    C[2] = z;
  }
};

/*!
 * \brief Adds to points. 
 */
Point3D operator+(Point3D A, const Point3D &B);

/*!
 * \brief Raw data structure used to hold the rotation matrix.
 */
struct RotationMatrix {
  double P[3][3];
};

/*!
 * \brief Dot product between a 3x3 matrix (RotationMatrix) and a 3x1 vector (Point3D).
 */
Point3D operator*(const RotationMatrix &A, const Point3D &B);

/*!
 * \brief Dot product between two 3x3 matrices (RotationMatrix).
 */
RotationMatrix operator*(const RotationMatrix &A, const RotationMatrix &B);

/*!
 * \brief Returns RotationMatrix for rotation around x-axis for theta radians. 
 */
RotationMatrix rotateX(double theta);

/*!
 * \brief Returns RotationMatrix for rotation around y-axis for theta radians. 
 */
RotationMatrix rotateY(double theta);

/*!
 * \brief Returns RotationMatrix for rotation around z-axis for theta radians. 
 */
RotationMatrix rotateZ(double theta);

/*!
 * \brief Returns RotationMatrix for rotation around all axes. Angles
 *        given in radians. 
 */
RotationMatrix createRotationMatrix(double thetaX, double thetaY, double thetaZ);

/*!
 * \brief Returns translation vector for three coordinates.
 */
Point3D getTranslation(double x, double y, double z);
