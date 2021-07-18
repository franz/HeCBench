/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, and Ari Harju

    This file is part of GPUQT.

    GPUQT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUQT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUQT.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "hamiltonian.h"
#include "model.h"
#include "vector.h"
#include <string.h>    // memcpy
#define BLOCK_SIZE 256 // optimized

#ifndef CPU_ONLY
void Hamiltonian::initialize_gpu(Model& model)
{
  n = model.number_of_atoms;
  max_neighbor = model.max_neighbor;
  energy_max = model.energy_max;
  grid_size = (model.number_of_atoms - 1) / BLOCK_SIZE + 1;

  q.submit([&] (handler &cgh) {
    auto acc = neighbor_number.get_access<sycl_discard_write>(cgh);
    cgh.copy(model.neighbor_number, acc);
  });

  int* neighbor_list_new = new int[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      neighbor_list_new[m * n + i] = model.neighbor_list[i * max_neighbor + m];
    }
  }

  q.submit([&] (handler &cgh) {
    auto acc = neighbor_list.get_access<sycl_discard_write>(cgh);
    cgh.copy(neighbor_list_new, acc);
  });

  q.submit([&] (handler &cgh) {
    auto acc = potential.get_access<sycl_discard_write>(cgh);
    cgh.copy(model.potential, acc);
  });

  real* hopping_real_new = new real[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      hopping_real_new[m * n + i] = model.hopping_real[i * max_neighbor + m];
    }
  }

  q.submit([&] (handler &cgh) {
    auto acc = hopping_real.get_access<sycl_discard_write>(cgh);
    cgh.copy(hopping_real_new, acc);
  });

  real* hopping_imag_new = new real[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      hopping_imag_new[m * n + i] = model.hopping_imag[i * max_neighbor + m];
    }
  }

  q.submit([&] (handler &cgh) {
    auto acc = hopping_imag.get_access<sycl_discard_write>(cgh);
    cgh.copy(hopping_imag_new, acc);
  });

  real* xx_new = new real[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      xx_new[m * n + i] = model.xx[i * max_neighbor + m];
    }
  }

  q.submit([&] (handler &cgh) {
    auto acc = xx.get_access<sycl_discard_write>(cgh);
    cgh.copy(xx_new, acc);
  });

  q.wait();

  delete[] model.neighbor_number;
  delete[] model.potential;
  delete[] model.neighbor_list;
  delete[] model.hopping_imag;
  delete[] model.hopping_real;
  delete[] model.xx;
  delete[] neighbor_list_new;
  delete[] hopping_real_new;
  delete[] hopping_imag_new;
  delete[] xx_new;

}
#else
void Hamiltonian::initialize_cpu(Model& model)
{
  n = model.number_of_atoms;
  max_neighbor = model.max_neighbor;
  energy_max = model.energy_max;
  int number_of_pairs = model.number_of_pairs;

  neighbor_number = new int[n];
  memcpy(neighbor_number, model.neighbor_number, sizeof(int) * n);
  delete[] model.neighbor_number;

  neighbor_list = new int[number_of_pairs];
  memcpy(neighbor_list, model.neighbor_list, sizeof(int) * number_of_pairs);
  delete[] model.neighbor_list;

  potential = new real[n];
  memcpy(potential, model.potential, sizeof(real) * n);
  delete[] model.potential;

  hopping_real = new real[number_of_pairs];
  memcpy(hopping_real, model.hopping_real, sizeof(real) * number_of_pairs);
  delete[] model.hopping_real;

  hopping_imag = new real[number_of_pairs];
  memcpy(hopping_imag, model.hopping_imag, sizeof(real) * number_of_pairs);
  delete[] model.hopping_imag;

  xx = new real[number_of_pairs];
  memcpy(xx, model.xx, sizeof(real) * number_of_pairs);
  delete[] model.xx;
}
#endif

Hamiltonian::Hamiltonian(Model& model)
#ifndef CPU_ONLY
: neighbor_number{model.number_of_atoms},
  neighbor_list {model.number_of_pairs},
  potential{model.number_of_atoms},
  hopping_real{model.number_of_pairs},
  hopping_imag{model.number_of_pairs},
  xx (model.number_of_pairs)
#endif

{
#ifndef CPU_ONLY
  initialize_gpu(model);
#else
  initialize_cpu(model);
#endif
}

Hamiltonian::~Hamiltonian()
{
#ifndef CPU_ONLY
#else
  delete[] neighbor_number;
  delete[] neighbor_list;
  delete[] potential;
  delete[] hopping_real;
  delete[] hopping_imag;
  delete[] xx;
#endif
}

#ifndef CPU_ONLY
void gpu_apply_hamiltonian(
  nd_item<1> &item,
  const int number_of_atoms,
  const real energy_max,
  const  int* __restrict g_neighbor_number,
  const  int* __restrict g_neighbor_list,
  const real* __restrict g_potential,
  const real* __restrict g_hopping_real,
  const real* __restrict g_hopping_imag,
  const real* __restrict g_state_in_real,
  const real* __restrict g_state_in_imag,
        real* __restrict g_state_out_real,
        real* __restrict g_state_out_imag)
{
  int n = item.get_global_id(0);
  if (n < number_of_atoms) {
    real temp_real = g_potential[n] * g_state_in_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_in_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale
    g_state_out_real[n] = temp_real;
    g_state_out_imag[n] = temp_imag;
  }
}
#else
void cpu_apply_hamiltonian(
  int number_of_atoms,
  int max_neighbor,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_state_in_real,
  real* g_state_in_imag,
  real* g_state_out_real,
  real* g_state_out_imag)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = g_potential[n] * g_state_in_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_in_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale
    g_state_out_real[n] = temp_real;
    g_state_out_imag[n] = temp_imag;
  }
}
#endif

// |output> = H |input>
void Hamiltonian::apply(Vector& input, Vector& output)
{
#ifndef CPU_ONLY
  range<1> gws (grid_size * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);
  const int size = n;
  const real emax = energy_max;

  q.submit([&] (handler &cgh) {
    auto neighbor_number_acc = neighbor_number.get_access<sycl_read>(cgh);
    auto neighbor_list_acc = neighbor_list.get_access<sycl_read>(cgh);
    auto potential_acc = potential.get_access<sycl_read>(cgh);
    auto hopping_real_acc = hopping_real.get_access<sycl_read>(cgh);
    auto hopping_imag_acc = hopping_imag.get_access<sycl_read>(cgh);
    auto input_real_part_acc = input.real_part.get_access<sycl_read>(cgh);
    auto input_imag_part_acc = input.imag_part.get_access<sycl_read>(cgh); 
    auto output_real_part_acc = output.real_part.get_access<sycl_discard_write>(cgh);
    auto output_imag_part_acc = output.imag_part.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class apply_hamiltonian>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {

      gpu_apply_hamiltonian(
        item,
        size, 
        emax, 
        neighbor_number_acc.get_pointer(), 
        neighbor_list_acc.get_pointer(),
        potential_acc.get_pointer(),
        hopping_real_acc.get_pointer(),
        hopping_imag_acc.get_pointer(),
        input_real_part_acc.get_pointer(), 
        input_imag_part_acc.get_pointer(), 
        output_real_part_acc.get_pointer(), 
        output_imag_part_acc.get_pointer());
    });
  });
#else
  cpu_apply_hamiltonian(
    n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
    hopping_imag, input.real_part, input.imag_part, output.real_part, output.imag_part);
#endif
}

#ifndef CPU_ONLY
void gpu_apply_commutator(
  nd_item<1> &item,
  const int number_of_atoms,
  const real energy_max,
  const  int* __restrict g_neighbor_number,
  const  int* __restrict g_neighbor_list,
  const real* __restrict g_hopping_real,
  const real* __restrict g_hopping_imag,
  const real* __restrict g_xx,
  const real* __restrict g_state_in_real,
  const real* __restrict g_state_in_imag,
        real* __restrict g_state_out_real,
        real* __restrict g_state_out_imag)
{
  int n = item.get_global_id(0);
  if (n < number_of_atoms) {
    real temp_real = 0.0;
    real temp_imag = 0.0;
    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      real xx = g_xx[index_1];
      temp_real -= (a * c - b * d) * xx;
      temp_imag -= (a * d + b * c) * xx;
    }
    g_state_out_real[n] = temp_real / energy_max; // scale
    g_state_out_imag[n] = temp_imag / energy_max; // scale
  }
}
#else
void cpu_apply_commutator(
  int number_of_atoms,
  int max_neighbor,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_xx,
  real* g_state_in_real,
  real* g_state_in_imag,
  real* g_state_out_real,
  real* g_state_out_imag)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = 0.0;
    real temp_imag = 0.0;
    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      real xx = g_xx[index_1];
      temp_real -= (a * c - b * d) * xx;
      temp_imag -= (a * d + b * c) * xx;
    }
    g_state_out_real[n] = temp_real / energy_max; // scale
    g_state_out_imag[n] = temp_imag / energy_max; // scale
  }
}
#endif

// |output> = [X, H] |input>
void Hamiltonian::apply_commutator(Vector& input, Vector& output)
{
#ifndef CPU_ONLY
  const int size = n;
  const real emax = energy_max;
  range<1> gws (grid_size * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);

  q.submit([&] (handler &cgh) {
    auto neighbor_number_acc = neighbor_number.get_access<sycl_read>(cgh);
    auto neighbor_list_acc = neighbor_list.get_access<sycl_read>(cgh);
    auto hopping_real_acc = hopping_real.get_access<sycl_read>(cgh);
    auto hopping_imag_acc = hopping_imag.get_access<sycl_read>(cgh);
    auto xx_acc = xx.get_access<sycl_read>(cgh);
    auto input_real_part_acc = input.real_part.get_access<sycl_read>(cgh);
    auto input_imag_part_acc = input.imag_part.get_access<sycl_read>(cgh);
    auto output_real_part_acc = output.real_part.get_access<sycl_discard_write>(cgh);
    auto output_imag_part_acc = output.imag_part.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class apply_comm>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gpu_apply_commutator(
        item,
        size,
        emax, 
        neighbor_number_acc.get_pointer(),
        neighbor_list_acc.get_pointer(),
        hopping_real_acc.get_pointer(),
        hopping_imag_acc.get_pointer(),
        xx_acc.get_pointer(),
        input_real_part_acc.get_pointer(),
        input_imag_part_acc.get_pointer(), 
        output_real_part_acc.get_pointer(), 
        output_imag_part_acc.get_pointer());
    });
  });
#else
  cpu_apply_commutator(
    n, max_neighbor, energy_max, neighbor_number, neighbor_list, hopping_real, hopping_imag, xx,
    input.real_part, input.imag_part, output.real_part, output.imag_part);
#endif
}

#ifndef CPU_ONLY
void gpu_apply_current(
  nd_item<1> &item,
  const int number_of_atoms,
  const int*__restrict g_neighbor_number,
  const int*__restrict g_neighbor_list,
  const real*__restrict g_hopping_real,
  const real*__restrict g_hopping_imag,
  const real*__restrict g_xx,
  const real*__restrict g_state_in_real,
  const real*__restrict g_state_in_imag,
  real*__restrict g_state_out_real,
  real*__restrict g_state_out_imag)
{
  int n = item.get_global_id(0);
  if (n < number_of_atoms) {
    real temp_real = 0.0;
    real temp_imag = 0.0;
    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      temp_real += (a * c - b * d) * g_xx[index_1];
      temp_imag += (a * d + b * c) * g_xx[index_1];
    }
    g_state_out_real[n] = +temp_imag;
    g_state_out_imag[n] = -temp_real;
  }
}
#else
void cpu_apply_current(
  const int number_of_atoms,
  const int max_neighbor,
  const  int* __restrict g_neighbor_number,
  const  int* __restrict g_neighbor_list,
  const real* __restrict g_hopping_real,
  const real* __restrict g_hopping_imag,
  const real* __restrict g_xx,
  const real* __restrict g_state_in_real,
  const real* __restrict g_state_in_imag,
        real* __restrict g_state_out_real,
        real* __restrict g_state_out_imag)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = 0.0;
    real temp_imag = 0.0;
    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      temp_real += (a * c - b * d) * g_xx[index_1];
      temp_imag += (a * d + b * c) * g_xx[index_1];
    }
    g_state_out_real[n] = +temp_imag;
    g_state_out_imag[n] = -temp_real;
  }
}
#endif

// |output> = V |input>
void Hamiltonian::apply_current(Vector& input, Vector& output)
{
#ifndef CPU_ONLY
  range<1> gws (grid_size * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);
  const int size = n;

  q.submit([&] (handler &cgh) {
    auto neighbor_number_acc = neighbor_number.get_access<sycl_read>(cgh);
    auto neighbor_list_acc = neighbor_list.get_access<sycl_read>(cgh);
    auto hopping_real_acc = hopping_real.get_access<sycl_read>(cgh);
    auto hopping_imag_acc = hopping_imag.get_access<sycl_read>(cgh);
    auto xx_acc = xx.get_access<sycl_read>(cgh);
    auto input_real_part_acc = input.real_part.get_access<sycl_read>(cgh);
    auto input_imag_part_acc = input.imag_part.get_access<sycl_read>(cgh);
    auto output_real_part_acc = output.real_part.get_access<sycl_discard_write>(cgh);
    auto output_imag_part_acc = output.imag_part.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class apply_current>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gpu_apply_current(
        item,
        size,
        neighbor_number_acc.get_pointer(),
        neighbor_list_acc.get_pointer(),
        hopping_real_acc.get_pointer(),
        hopping_imag_acc.get_pointer(),
        xx_acc.get_pointer(),
        input_real_part_acc.get_pointer(),
        input_imag_part_acc.get_pointer(), 
        output_real_part_acc.get_pointer(), 
        output_imag_part_acc.get_pointer());
    });
  });
#else
  cpu_apply_current(
    n, max_neighbor, neighbor_number, neighbor_list, hopping_real, hopping_imag, xx,
    input.real_part, input.imag_part, output.real_part, output.imag_part);
#endif
}

// Kernel which calculates the two first terms of time evolution as described by
// Eq. (36) in [Comput. Phys. Commun.185, 28 (2014)].
#ifndef CPU_ONLY
void gpu_chebyshev_01(
  nd_item<1> &item,
  const int number_of_atoms,
  const real*__restrict g_state_0_real,
  const real*__restrict g_state_0_imag,
  const real*__restrict g_state_1_real,
  const real*__restrict g_state_1_imag,
        real*__restrict g_state_real,
        real*__restrict g_state_imag,
  const real b0,
  const real b1,
  const int direction)
{
  int n = item.get_global_id(0);
  if (n < number_of_atoms) {
    real bessel_0 = b0;
    real bessel_1 = b1 * direction;
    g_state_real[n] = bessel_0 * g_state_0_real[n] + bessel_1 * g_state_1_imag[n];
    g_state_imag[n] = bessel_0 * g_state_0_imag[n] - bessel_1 * g_state_1_real[n];
  }
}
#else
void cpu_chebyshev_01(
  int number_of_atoms,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_real,
  real* g_state_imag,
  real b0,
  real b1,
  int direction)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real bessel_0 = b0;
    real bessel_1 = b1 * direction;
    g_state_real[n] = bessel_0 * g_state_0_real[n] + bessel_1 * g_state_1_imag[n];
    g_state_imag[n] = bessel_0 * g_state_0_imag[n] - bessel_1 * g_state_1_real[n];
  }
}
#endif

// Wrapper for the kernel above
void Hamiltonian::chebyshev_01(
  Vector& state_0, Vector& state_1, Vector& state, real bessel_0, real bessel_1, int direction)
{
#ifndef CPU_ONLY
  range<1> gws (grid_size * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);
  const int size = n;

  q.submit([&] (handler &cgh) {
    auto input0_real_part_acc = state_0.real_part.get_access<sycl_read>(cgh);
    auto input0_imag_part_acc = state_0.imag_part.get_access<sycl_read>(cgh);
    auto input1_real_part_acc = state_1.real_part.get_access<sycl_read>(cgh);
    auto input1_imag_part_acc = state_1.imag_part.get_access<sycl_read>(cgh);
    auto output_real_part_acc = state.real_part.get_access<sycl_discard_write>(cgh);
    auto output_imag_part_acc = state.imag_part.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class chebyshev01>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gpu_chebyshev_01(
        item,
        size,
        input0_real_part_acc.get_pointer(),
        input0_imag_part_acc.get_pointer(), 
        input1_real_part_acc.get_pointer(),
        input1_imag_part_acc.get_pointer(), 
        output_real_part_acc.get_pointer(), 
        output_imag_part_acc.get_pointer(),
        bessel_0, bessel_1, direction);
    });
  });
#else
  cpu_chebyshev_01(
    n, state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part, state.real_part,
    state.imag_part, bessel_0, bessel_1, direction);
#endif
}

// Kernel for calculating further terms of Eq. (36)
// in [Comput. Phys. Commun.185, 28 (2014)].
#ifndef CPU_ONLY
void gpu_chebyshev_2(
  nd_item<1> &item,
  const int number_of_atoms,
  const real energy_max,
  const  int* __restrict g_neighbor_number,
  const  int* __restrict g_neighbor_list,
  const real* __restrict g_potential,
  const real* __restrict g_hopping_real,
  const real* __restrict g_hopping_imag,
  const real* __restrict g_state_0_real,
  const real* __restrict g_state_0_imag,
  const real* __restrict g_state_1_real,
  const real* __restrict g_state_1_imag,
        real* __restrict g_state_2_real,
        real* __restrict g_state_2_imag,
        real* __restrict g_state_real,
        real* __restrict g_state_imag,
  const real bessel_m,
  const int label)
{
  int n = item.get_global_id(0);
  if (n < number_of_atoms) {
    real temp_real = g_potential[n] * g_state_1_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale

    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    switch (label) {
      case 1: {
        g_state_real[n] += bessel_m * temp_real;
        g_state_imag[n] += bessel_m * temp_imag;
        break;
      }
      case 2: {
        g_state_real[n] -= bessel_m * temp_real;
        g_state_imag[n] -= bessel_m * temp_imag;
        break;
      }
      case 3: {
        g_state_real[n] += bessel_m * temp_imag;
        g_state_imag[n] -= bessel_m * temp_real;
        break;
      }
      case 4: {
        g_state_real[n] -= bessel_m * temp_imag;
        g_state_imag[n] += bessel_m * temp_real;
        break;
      }
    }
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;
  }
}
#else
void cpu_chebyshev_2(
  int number_of_atoms,
  int max_neighbor,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_2_real,
  real* g_state_2_imag,
  real* g_state_real,
  real* g_state_imag,
  real bessel_m,
  int label)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = g_potential[n] * g_state_1_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale

    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    switch (label) {
      case 1: {
        g_state_real[n] += bessel_m * temp_real;
        g_state_imag[n] += bessel_m * temp_imag;
        break;
      }
      case 2: {
        g_state_real[n] -= bessel_m * temp_real;
        g_state_imag[n] -= bessel_m * temp_imag;
        break;
      }
      case 3: {
        g_state_real[n] += bessel_m * temp_imag;
        g_state_imag[n] -= bessel_m * temp_real;
        break;
      }
      case 4: {
        g_state_real[n] -= bessel_m * temp_imag;
        g_state_imag[n] += bessel_m * temp_real;
        break;
      }
    }
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;
  }
}
#endif

// Wrapper for the kernel above
void Hamiltonian::chebyshev_2(
  Vector& state_0, Vector& state_1, Vector& state_2, Vector& state, real bessel_m, int label)
{
#ifndef CPU_ONLY
  //gpu_chebyshev_2<<<grid_size, BLOCK_SIZE>>>(
    //n, energy_max, neighbor_number, neighbor_list, potential, hopping_real, hopping_imag,
    //state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part, state_2.real_part,
    //state_2.imag_part, state.real_part, state.imag_part, bessel_m, label);

  range<1> gws (grid_size * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);
  const int size = n;
  const real emax = energy_max;

  q.submit([&] (handler &cgh) {
    auto neighbor_number_acc = neighbor_number.get_access<sycl_read>(cgh);
    auto neighbor_list_acc = neighbor_list.get_access<sycl_read>(cgh);
    auto potential_acc = potential.get_access<sycl_read>(cgh);
    auto hopping_real_acc = hopping_real.get_access<sycl_read>(cgh);
    auto hopping_imag_acc = hopping_imag.get_access<sycl_read>(cgh);
    auto state0_real_part_acc = state_0.real_part.get_access<sycl_read>(cgh);
    auto state0_imag_part_acc = state_0.imag_part.get_access<sycl_read>(cgh);
    auto state1_real_part_acc = state_1.real_part.get_access<sycl_read>(cgh);
    auto state1_imag_part_acc = state_1.imag_part.get_access<sycl_read>(cgh);
    auto state2_real_part_acc = state_2.real_part.get_access<sycl_discard_write>(cgh);
    auto state2_imag_part_acc = state_2.imag_part.get_access<sycl_discard_write>(cgh);
    auto state_real_part_acc = state.real_part.get_access<sycl_read_write>(cgh);
    auto state_imag_part_acc = state.imag_part.get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class chebyshev02>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gpu_chebyshev_2(
        item,
        size,
        emax, 
        neighbor_number_acc.get_pointer(),
        neighbor_list_acc.get_pointer(),
        potential_acc.get_pointer(),
        hopping_real_acc.get_pointer(),
        hopping_imag_acc.get_pointer(),
        state0_real_part_acc.get_pointer(),
        state0_imag_part_acc.get_pointer(), 
        state1_real_part_acc.get_pointer(),
        state1_imag_part_acc.get_pointer(), 
        state2_real_part_acc.get_pointer(),
        state2_imag_part_acc.get_pointer(), 
        state_real_part_acc.get_pointer(), 
        state_imag_part_acc.get_pointer(),
        bessel_m, label);
    });
  });
#else
  cpu_chebyshev_2(
    n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
    hopping_imag, state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part,
    state_2.real_part, state_2.imag_part, state.real_part, state.imag_part, bessel_m, label);
#endif
}

// Kernel which calculates the two first terms of commutator [X, U(dt)]
// Corresponds to Eq. (37) in [Comput. Phys. Commun.185, 28 (2014)].
#ifndef CPU_ONLY
void gpu_chebyshev_1x(
  nd_item<1> &item,
  const int number_of_atoms,
  const real*__restrict g_state_1x_real,
  const real*__restrict g_state_1x_imag,
        real*__restrict g_state_real,
        real*__restrict g_state_imag,
  const real g_bessel_1)
{
  int n = item.get_global_id(0);
  if (n < number_of_atoms) {
    real b1 = g_bessel_1;
    g_state_real[n] = +b1 * g_state_1x_imag[n];
    g_state_imag[n] = -b1 * g_state_1x_real[n];
  }
}
#else
void cpu_chebyshev_1x(
  int number_of_atoms,
  real* g_state_1x_real,
  real* g_state_1x_imag,
  real* g_state_real,
  real* g_state_imag,
  real g_bessel_1)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real b1 = g_bessel_1;
    g_state_real[n] = +b1 * g_state_1x_imag[n];
    g_state_imag[n] = -b1 * g_state_1x_real[n];
  }
}
#endif

// Wrapper for kernel above
void Hamiltonian::chebyshev_1x(Vector& input, Vector& output, real bessel_1)
{
#ifndef CPU_ONLY
  //gpu_chebyshev_1x<<<grid_size, BLOCK_SIZE>>>(
  //  n, input.real_part, input.imag_part, output.real_part, output.imag_part, bessel_1);
  //CHECK(cudaGetLastError());

  range<1> gws (grid_size * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);
  const int size = n;

  q.submit([&] (handler &cgh) {
    auto input_real_part_acc = input.real_part.get_access<sycl_read>(cgh);
    auto input_imag_part_acc = input.imag_part.get_access<sycl_read>(cgh);
    auto output_real_part_acc = output.real_part.get_access<sycl_discard_write>(cgh);
    auto output_imag_part_acc = output.imag_part.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class chebyshev_1x>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gpu_chebyshev_1x(
        item,
        size,
        input_real_part_acc.get_pointer(),
        input_imag_part_acc.get_pointer(),
        output_real_part_acc.get_pointer(),
        output_imag_part_acc.get_pointer(),
        bessel_1);
    });
  });
#else
  cpu_chebyshev_1x(n, input.real_part, input.imag_part, output.real_part, output.imag_part, bessel_1);
#endif
}

// Kernel which calculates the further terms of [X, U(dt)]
#ifndef CPU_ONLY
void gpu_chebyshev_2x(
  nd_item<1> &item,
  const int number_of_atoms,
  const real energy_max,
  const  int* __restrict g_neighbor_number,
  const  int* __restrict g_neighbor_list,
  const real* __restrict g_potential,
  const real* __restrict g_hopping_real,
  const real* __restrict g_hopping_imag,
  const real* __restrict g_xx,
  const real* __restrict g_state_0_real,
  const real* __restrict g_state_0_imag,
  const real* __restrict g_state_0x_real,
  const real* __restrict g_state_0x_imag,
  const real* __restrict g_state_1_real,
  const real* __restrict g_state_1_imag,
  const real* __restrict g_state_1x_real,
  const real* __restrict g_state_1x_imag,
        real* __restrict g_state_2_real,
        real* __restrict g_state_2_imag,
        real* __restrict g_state_2x_real,
        real* __restrict g_state_2x_imag,
        real* __restrict g_state_real,
        real* __restrict g_state_imag,
  const real g_bessel_m,
  const int g_label)
{
  int n = item.get_global_id(0);
  if (n < number_of_atoms) {
    real temp_real = g_potential[n] * g_state_1_real[n];    // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n];    // on-site
    real temp_x_real = g_potential[n] * g_state_1x_real[n]; // on-site
    real temp_x_imag = g_potential[n] * g_state_1x_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];

      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping

      real cx = g_state_1x_real[index_2];
      real dx = g_state_1x_imag[index_2];
      temp_x_real += a * cx - b * dx; // hopping
      temp_x_imag += a * dx + b * cx; // hopping

      real xx = g_xx[index_1];
      temp_x_real -= (a * c - b * d) * xx; // hopping
      temp_x_imag -= (a * d + b * c) * xx; // hopping
    }

    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale
    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;

    temp_x_real /= energy_max; // scale
    temp_x_imag /= energy_max; // scale
    temp_x_real = 2.0 * temp_x_real - g_state_0x_real[n];
    temp_x_imag = 2.0 * temp_x_imag - g_state_0x_imag[n];
    g_state_2x_real[n] = temp_x_real;
    g_state_2x_imag[n] = temp_x_imag;

    real bessel_m = g_bessel_m;
    switch (g_label) {
      case 1: {
        g_state_real[n] += bessel_m * temp_x_real;
        g_state_imag[n] += bessel_m * temp_x_imag;
        break;
      }
      case 2: {
        g_state_real[n] -= bessel_m * temp_x_real;
        g_state_imag[n] -= bessel_m * temp_x_imag;
        break;
      }
      case 3: {
        g_state_real[n] += bessel_m * temp_x_imag;
        g_state_imag[n] -= bessel_m * temp_x_real;
        break;
      }
      case 4: {
        g_state_real[n] -= bessel_m * temp_x_imag;
        g_state_imag[n] += bessel_m * temp_x_real;
        break;
      }
    }
  }
}
#else
void cpu_chebyshev_2x(
  int number_of_atoms,
  int max_neighbor,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_xx,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_0x_real,
  real* g_state_0x_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_1x_real,
  real* g_state_1x_imag,
  real* g_state_2_real,
  real* g_state_2_imag,
  real* g_state_2x_real,
  real* g_state_2x_imag,
  real* g_state_real,
  real* g_state_imag,
  real g_bessel_m,
  int g_label)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = g_potential[n] * g_state_1_real[n];    // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n];    // on-site
    real temp_x_real = g_potential[n] * g_state_1x_real[n]; // on-site
    real temp_x_imag = g_potential[n] * g_state_1x_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];

      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping

      real cx = g_state_1x_real[index_2];
      real dx = g_state_1x_imag[index_2];
      temp_x_real += a * cx - b * dx; // hopping
      temp_x_imag += a * dx + b * cx; // hopping

      real xx = g_xx[index_1];
      temp_x_real -= (a * c - b * d) * xx; // hopping
      temp_x_imag -= (a * d + b * c) * xx; // hopping
    }

    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale
    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;

    temp_x_real /= energy_max; // scale
    temp_x_imag /= energy_max; // scale
    temp_x_real = 2.0 * temp_x_real - g_state_0x_real[n];
    temp_x_imag = 2.0 * temp_x_imag - g_state_0x_imag[n];
    g_state_2x_real[n] = temp_x_real;
    g_state_2x_imag[n] = temp_x_imag;

    real bessel_m = g_bessel_m;
    switch (g_label) {
      case 1: {
        g_state_real[n] += bessel_m * temp_x_real;
        g_state_imag[n] += bessel_m * temp_x_imag;
        break;
      }
      case 2: {
        g_state_real[n] -= bessel_m * temp_x_real;
        g_state_imag[n] -= bessel_m * temp_x_imag;
        break;
      }
      case 3: {
        g_state_real[n] += bessel_m * temp_x_imag;
        g_state_imag[n] -= bessel_m * temp_x_real;
        break;
      }
      case 4: {
        g_state_real[n] -= bessel_m * temp_x_imag;
        g_state_imag[n] += bessel_m * temp_x_real;
        break;
      }
    }
  }
}
#endif

// Wrapper for the kernel above
void Hamiltonian::chebyshev_2x(
  Vector& state_0,
  Vector& state_0x,
  Vector& state_1,
  Vector& state_1x,
  Vector& state_2,
  Vector& state_2x,
  Vector& state,
  real bessel_m,
  int label)
{
#ifndef CPU_ONLY
  //gpu_chebyshev_2x<<<grid_size, BLOCK_SIZE>>>(
    //n, energy_max, neighbor_number, neighbor_list, potential, hopping_real, hopping_imag, xx,
    //state_0.real_part, state_0.imag_part, state_0x.real_part, state_0x.imag_part, state_1.real_part,
    //state_1.imag_part, state_1x.real_part, state_1x.imag_part, state_2.real_part, state_2.imag_part,
    //state_2x.real_part, state_2x.imag_part, state.real_part, state.imag_part, bessel_m, label);

  range<1> gws (grid_size * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);
  const int size = n;
  const real emax = energy_max;

  q.submit([&] (handler &cgh) {
    auto neighbor_number_acc = neighbor_number.get_access<sycl_read>(cgh);
    auto neighbor_list_acc = neighbor_list.get_access<sycl_read>(cgh);
    auto potential_acc = potential.get_access<sycl_read>(cgh);
    auto hopping_real_acc = hopping_real.get_access<sycl_read>(cgh);
    auto hopping_imag_acc = hopping_imag.get_access<sycl_read>(cgh);
    auto xx_acc = xx.get_access<sycl_read>(cgh);
    auto state0_real_part_acc = state_0.real_part.get_access<sycl_read>(cgh);
    auto state0_imag_part_acc = state_0.imag_part.get_access<sycl_read>(cgh);
    auto state0x_real_part_acc = state_0x.real_part.get_access<sycl_read>(cgh);
    auto state0x_imag_part_acc = state_0x.imag_part.get_access<sycl_read>(cgh);
    auto state1_real_part_acc = state_1.real_part.get_access<sycl_read>(cgh);
    auto state1_imag_part_acc = state_1.imag_part.get_access<sycl_read>(cgh);
    auto state1x_real_part_acc = state_1x.real_part.get_access<sycl_read>(cgh);
    auto state1x_imag_part_acc = state_1x.imag_part.get_access<sycl_read>(cgh);
    auto state2_real_part_acc = state_2.real_part.get_access<sycl_discard_write>(cgh);
    auto state2_imag_part_acc = state_2.imag_part.get_access<sycl_discard_write>(cgh);
    auto state2x_real_part_acc = state_2x.real_part.get_access<sycl_discard_write>(cgh);
    auto state2x_imag_part_acc = state_2x.imag_part.get_access<sycl_discard_write>(cgh);
    auto state_real_part_acc = state.real_part.get_access<sycl_read_write>(cgh);
    auto state_imag_part_acc = state.imag_part.get_access<sycl_read_write>(cgh);

    cgh.parallel_for<class chebyshev2x>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gpu_chebyshev_2x(
        item,
        size, 
        emax, 
        neighbor_number_acc.get_pointer(),
        neighbor_list_acc.get_pointer(),
        potential_acc.get_pointer(),
        hopping_real_acc.get_pointer(),
        hopping_imag_acc.get_pointer(),
        xx_acc.get_pointer(),
        state0_real_part_acc.get_pointer(),
        state0_imag_part_acc.get_pointer(), 
        state0x_real_part_acc.get_pointer(),
        state0x_imag_part_acc.get_pointer(), 
        state1_real_part_acc.get_pointer(),
        state1_imag_part_acc.get_pointer(), 
        state1x_real_part_acc.get_pointer(),
        state1x_imag_part_acc.get_pointer(), 
        state2_real_part_acc.get_pointer(),
        state2_imag_part_acc.get_pointer(), 
        state2x_real_part_acc.get_pointer(),
        state2x_imag_part_acc.get_pointer(), 
        state_real_part_acc.get_pointer(), 
        state_imag_part_acc.get_pointer(),
        bessel_m, label);
    });
  });

#else
  cpu_chebyshev_2x(
    n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
    hopping_imag, xx, state_0.real_part, state_0.imag_part, state_0x.real_part, state_0x.imag_part,
    state_1.real_part, state_1.imag_part, state_1x.real_part, state_1x.imag_part, state_2.real_part,
    state_2.imag_part, state_2x.real_part, state_2x.imag_part, state.real_part, state.imag_part,
    bessel_m, label);
#endif
}

// Kernel for doing the Chebyshev iteration phi_2 = 2 * H * phi_1 - phi_0.
#ifndef CPU_ONLY
void gpu_kernel_polynomial(
  nd_item<1> &item,
  const int number_of_atoms,
  const real energy_max,
  const  int* __restrict g_neighbor_number,
  const  int* __restrict g_neighbor_list,
  const real* __restrict g_potential,
  const real* __restrict g_hopping_real,
  const real* __restrict g_hopping_imag,
  const real* __restrict g_state_0_real,
  const real* __restrict g_state_0_imag,
  const real* __restrict g_state_1_real,
  const real* __restrict g_state_1_imag,
        real* __restrict g_state_2_real,
        real* __restrict g_state_2_imag)
{
  int n = item.get_global_id(0);
  if (n < number_of_atoms) {
    real temp_real = g_potential[n] * g_state_1_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }

    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale

    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;
  }
}
#else
void cpu_kernel_polynomial(
  int number_of_atoms,
  int max_neighbor,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_2_real,
  real* g_state_2_imag)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = g_potential[n] * g_state_1_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }

    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale

    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;
  }
}
#endif

// Wrapper for the Chebyshev iteration
void Hamiltonian::kernel_polynomial(Vector& state_0, Vector& state_1, Vector& state_2)
{
#ifndef CPU_ONLY
  //gpu_kernel_polynomial<<<grid_size, BLOCK_SIZE>>>(
  //  n, energy_max, neighbor_number, neighbor_list, potential, hopping_real, hopping_imag,
  //  state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part, state_2.real_part,
  //  state_2.imag_part);

  range<1> gws (grid_size * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);
  const int size = n;
  const real emax = energy_max;

  q.submit([&] (handler &cgh) {
    auto neighbor_number_acc = neighbor_number.get_access<sycl_read>(cgh);
    auto neighbor_list_acc = neighbor_list.get_access<sycl_read>(cgh);
    auto potential_acc = potential.get_access<sycl_read>(cgh);
    auto hopping_real_acc = hopping_real.get_access<sycl_read>(cgh);
    auto hopping_imag_acc = hopping_imag.get_access<sycl_read>(cgh);
    auto state0_real_part_acc = state_0.real_part.get_access<sycl_read>(cgh);
    auto state0_imag_part_acc = state_0.imag_part.get_access<sycl_read>(cgh);
    auto state1_real_part_acc = state_1.real_part.get_access<sycl_read>(cgh);
    auto state1_imag_part_acc = state_1.imag_part.get_access<sycl_read>(cgh);
    auto state2_real_part_acc = state_2.real_part.get_access<sycl_discard_write>(cgh);
    auto state2_imag_part_acc = state_2.imag_part.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class polynomial>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gpu_kernel_polynomial(
        item,
        size,
        emax, 
        neighbor_number_acc.get_pointer(),
        neighbor_list_acc.get_pointer(),
        potential_acc.get_pointer(),
        hopping_real_acc.get_pointer(),
        hopping_imag_acc.get_pointer(),
        state0_real_part_acc.get_pointer(),
        state0_imag_part_acc.get_pointer(), 
        state1_real_part_acc.get_pointer(),
        state1_imag_part_acc.get_pointer(), 
        state2_real_part_acc.get_pointer(),
        state2_imag_part_acc.get_pointer());
    });
  });
#else
  cpu_kernel_polynomial(
    n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
    hopping_imag, state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part,
    state_2.real_part, state_2.imag_part);
#endif
}
