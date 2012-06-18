#pragma once
#include <cstdio>
#include <string>
#include <iostream>

#include "cuda_runtime.h"
#include"rotate.h"

namespace nbrahle { 
  /*!
   * \brief Used to hold a single Molecule. 
   */
  class Molecule
  {
   public:
    /*!
     * \brief Default constructor. 
     */
	  Molecule();
    /*!
     * \brief Copy constructor. 
     */
    Molecule(const Molecule&);
    /*!
     * \brief Used to construct molecule from raw data.
     */
    Molecule(const Point3D&, int, int);
    /*!
     * \brief Destrutor. 
     */
	  ~Molecule();

    /*!
     * \brief Getter for position of the molecule on the x-axis.
     */
    double x() const;
    /*!
     * \brief Getter for position of the molecule on the y-axis. 
     */
    double y() const;
    /*!
     * \brief Getter for position of the molecule on the z-axis.
     */
    double z() const;
    /*!
     * \brief Getter for cost of deletion. Deprecated.
     */
    double deletion_cost() const;
    /*!
     * \brief Getter for name of the molcule. 
     */
    int name() const { return name_; }
    /*!
     * \brief Getter for position of the molecule in the Protein. 
     */
    int position() const { return position_; }
    /*!
     * \brief Setter for position of the molecule in the Protein. 
     */
    void set_position(int position) { position_ = position; }
    /*!
     * \brief Setter for position of the molecule on the x-axis.
     */
    void set_x(double x) { x_ = x; }
    /*!
     * \brief Setter for position of the molecule on the y-axis.
     */
    void set_y(double y) { y_ = y; }
    /*!
     * \brief Setter for position of the molecule on the z-axis.
     */
    void set_z(double z) { z_ = z; }

    /*!
     * \brief Reads the molecule from std::istream. 
     */
    friend std::istream& operator>>(std::istream&, Molecule&);
    /*!
     * \brief Outputs the molecule to std::ostream.
     */
    friend std::ostream& operator<<(std::ostream&, Molecule&);

   private:
    double x_, y_, z_;
	  double deletion_cost_;
    int name_;
	  int position_;
  };
}
