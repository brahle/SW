#pragma once

#include "utilities.h"
#include "consts.h"

/*!
  * \brief Gives the temperature used in simulated annealing in k-th step.
  */
double temperature(int k) {
  return T0 * power(T1, k);
}

/*!
  * \brief Returns probabilty of accepting the new state, based on the energy of
  *        the current and the neighbouring state.
  */
double P(double old_e, double new_e, double T) {
  if (new_e < old_e) return 1.0;
  return 1 / (1 + exp((new_e - old_e) / T));
}

/*!
 * \brief Simulated Annealing algorithm. 
 */
template< typename StateType, typename EnergyFunction, typename NextFunction >
StateType annealing(const StateType &original,  //!< Starting state.
                    EnergyFunction getEnergy,   //!< Function that accepts a state and gets its energy.
                    NextFunction neighbour,     //!< Function that accepts a state and returns a neighbour.
                    int maxEvolution,           //!< Limit on the number of evolutions.
                    double acceptableEnergy,    //!< Energy limit that is considered acceptible.
                    int poolSize=20             //!< Number of neigbours tested in every step.
                    ) {
  StateType current_state = original, *next_states = new StateType[poolSize];
  double current_energy = getEnergy(current_state), *next_energies = new double[poolSize];
  int next_best;

  StateType best_state = current_state;
  double best_energy = current_energy;

  int reset_temperature = 0;
  int promjena_pred = 0;

  for (int i = 0; i < maxEvolution; ++i) {
    if (best_energy <= acceptableEnergy) break;
    double T = temperature(i-reset_temperature);

    next_best = 0;
    if (rand() < 0.1 * RAND_MAX) {
      next_states[0] = neighbour(current_state);
      next_energies[0] = getEnergy(next_states[0]);
    } else {
      for (int j = 0; j < poolSize; ++j) {
        next_states[j] = neighbour(current_state);
        next_energies[j] = getEnergy(next_states[j]);
        if (next_energies[j] < next_energies[next_best]) {
          next_best = j;
        }
      }
    }
    printf("%d. Best = %g; Curr = %g; Neigh = %g; Temp = %g\n", i+1, best_energy, current_energy, next_energies[next_best], T);

    double a = P(current_energy, next_energies[next_best], T);
    double b = rand();
    //printf("S %g na %g uz temperaturu %g: vjerojatnost je %g\n", current_energy, next_energies[next_best], T, a*100.0);
    if (a*RAND_MAX > b) {
      current_state = next_states[next_best];
      current_energy = next_energies[next_best];
      promjena_pred = 0;
    } else {
      ++promjena_pred;
      if (promjena_pred == 15) {
        reset_temperature = i+1;
        promjena_pred = 0;
      }
    }
    if (current_energy < best_energy) {
      best_state = current_state;
      best_energy = current_energy;
    }

  }

  printf("Konacna najbolja energija = %g\n", best_energy);
  delete [] next_states;
  delete [] next_energies;
  return best_state;
}
