#pragma once
#include <algorithm>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"

#include "Molecule.h"
#include "utilities.h"
#include "rotate.h"

namespace nbrahle { 
  /*!
   * \brief Used to hold a single Protein. 
   */
  class Protein {
  public:
    /*!
     * \brief Default constructor.
     */
	  Protein(void);
    /*!
     * \brief Constructor from an array of molecules.
     */
    Protein(int, Molecule*, bool);
    /*!
     * \brief Copy constructor.
     */
    Protein(const Protein&);
    /*!
     * \brief Constructor from raw data.
     */
    Protein(const std::vector< Point3D >&);
    /*!
     * \brief Destructor.
     */
	  ~Protein(void);

    /*!
     * \brief Getter for the number of molecules in this protein. 
     */
    __device__ __host__ int n() const;
    /*!
     * \brief Getter for the last interesting molecule in this protein. Deprecated.
     *
     * A molecule is interesting if it can potentially be in the best allignment.
     */
    int end() const { return end_; }
    /*!
     * \brief Getter for i-th molecule in this protein. 
     */
    __device__ __host__ Molecule& operator[] (int i);
    /*!
     * \brief Getter for i-th molecule in this protein. 
     */
    __device__ __host__ Molecule& operator[] (int i) const;
    /*!
     * \brief Getter for the array of all molecules in this protein.. 
     */
    Molecule* molekule() const { return molecules_; }

    /*!
     * \brief Setter for the end of the protein. Deprecated.
     */
    void SetEnd(int end) { end_ = end; }

    /*!
     * \brief Creates the copy of the molecule on CUDA.
     */
    Protein createCopyOnCuda() const;
    /*!
     * \brief Reverses the protein.
     */
    void Reverse();
    /*!
     * \brief Resizes the protein to new size.
     */
    void Resize(int newSize);

    /*!
     * \brief Copies x-axes coordinates of every molecule in this protein to device.
     */
    double* CopyXToDevice() const;
    /*!
     * \brief Copies y-axes coordinates of every molecule in this protein to device.
     */
    double* CopyYToDevice() const;
    /*!
     * \brief Copies z-axes coordinates of every molecule in this protein to device.
     */
    double* CopyZToDevice() const;
    /*!
     * \brief Copies deletion cost of every molecule in this protein to device.
     *        Deprecated.
     */
    double* CopyDCToDevice() const;

    /*!
     * \brief Reads the protein from std::istream. 
     */
    friend std::istream& operator>>(std::istream&, nbrahle::Protein&);
    /*!
     * \brief Outputs the protein to std::ostream. 
     */
    friend std::ostream& operator<<(std::ostream&, nbrahle::Protein&);

   private:
    int n_;
    Molecule* molecules_;
    bool on_cuda_;
    int end_;
  };
}
