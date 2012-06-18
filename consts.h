#pragma once

const double GAP_START = -15.0;   //!< Cost of starting a gap
const double GAP_CONTINUE = -5.0; //!< Cost of expanding an existing gap

const int ANNEALING_STEPS = 20;   //!< Number of evoluations for simulated annealing.
const int POOL_SIZE = 20;         //!< Size of pool used by simulated annealing.
const double T0 = 10000.0;        //!< Starting temperature.
const double T1 = 0.9;            //!< Cooldown factor.

/*!
 * \brief Minimal number of elements in the array needed for findMaximum to
 *        start using reduceMax.
 */
const int FIND_MAXIMUM_PARALLEL_CUTOFF = 1<<14;

const int PDB_ATOM_RES_NO_LEN = 1000;
const int PDB_ATOM_RES_NAME_LEN = 20;
const int MAX_NO_ATOMS = 20000;
const int PDB_ATOM_ATOM_NAME_LEN = 20;
const int MEDSTRING = 1000;
