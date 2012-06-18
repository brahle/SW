#pragma once

#include <vector>

#include "external_structs.h"
#include "rotate.h"

/*!
 * \brief
 */
class State {
public:
  std::vector< Point3D > original, P;
  double dx, dy, dz;
  double thetaX, thetaY, thetaZ;

  /*!
   * \brief Default constructor.
   */
  State();
  /*!
   * \brief Constructor from a Protein data-structure.
   */
  State(::Protein *);

  /*!
   * \brief Reads data from a string.
   */
  void read(const char *);
  /*!
   * \brief Translates and rotates itself. 
   */
  void transformMyself(double, double);
  /*!
   * \brief Copies the other state and then transforms (using translation and rotation) itself.
   */
  void transformOther(const State &);
  /*!
   * \brief Resets the rotations and translations. 
   */
  void reset();
};
