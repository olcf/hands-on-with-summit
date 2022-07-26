/*---------------------------------------------------------------------------*/
/*!
 * \file   quantities_testing_kernels.h
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  quantities_testing, code for comp. kernel.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _quantities_testing_kernels_h_
#define _quantities_testing_kernels_h_

#include "types_kernels.h"
#include "dimensions_kernels.h"
#include "array_accessors_kernels.h"
#include "pointer_kernels.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Type of boundary conditions---*/

TARGET_HD static inline Bool_t Quantities_bc_vacuum()
{
  return Bool_false;
}

/*===========================================================================*/
/*---Struct to hold pointers to arrays associated with physical quantities---*/

typedef struct
{
  Pointer  a_from_m;
  Pointer  m_from_a;
  int*     ix_base_vals;
  int*     iy_base_vals;
  int      ix_base;
  int      iy_base;
  int      ncell_x_g;
  int      ncell_y_g;
  int      ncell_z_g;
} Quantities;

/*===========================================================================*/
/*---Scale factor for energy---*/
/*---pseudo-private member function---*/

TARGET_HD static inline int Quantities_scalefactor_energy_( int ie,
                                                            Dimensions dims )
{
  /*---Random power-of-two multiplier for each energy group,
       to help catch errors regarding indexing of energy groups.
  ---*/
  Assert( ie >= 0 && ie < dims.ne );

  const int im = 714025;
  const int ia = 1366;
  const int ic = 150889;

  int result = ( (ie)*ia + ic ) % im;
  result = result & ( (1<<2) - 1 );
  result = 1 << result;

  return result;
}

/*===========================================================================*/
/*---Scale factor for unknown---*/
/*---pseudo-private member function---*/

TARGET_HD static inline int Quantities_scalefactor_unknown_( int iu )
{
  /*---Random power-of-two multiplier for each cell unknown,
       to help catch errors regarding indexing of cell unknowns.
  ---*/
  Assert( iu >= 0 && iu < NU );

  const int im = 312500;
  const int ia = 741;
  const int ic = 66037;

  int result = ( (iu)*ia + ic ) % im;
  result = result & ( (1<<2) - 1 );
  result = 1 << result;

  return result;
}

/*===========================================================================*/
/*---Scale factor for space---*/
/*---pseudo-private member function---*/

TARGET_HD static inline int Quantities_scalefactor_space_(
                                                  const Quantities* quan,
                                                  int ix_g,
                                                  int iy_g,
                                                  int iz_g )
{
  /*---Create power of 2 based on hash of the spatial location.
  ---*/
  Assert( ix_g >= -1 && ix_g <= quan->ncell_x_g );
  Assert( iy_g >= -1 && iy_g <= quan->ncell_y_g );
  Assert( iz_g >= -1 && iz_g <= quan->ncell_z_g );

  int result = 0;

#ifndef RELAXED_TESTING
  const int im = 134456;
  const int ia = 8121;
  const int ic = 28411;

  result = ( (result+(ix_g+2))*ia + ic ) % im;
  result = ( (result+(iy_g+2))*ia + ic ) % im;
  result = ( (result+(iz_g+2))*ia + ic ) % im;
  result = ( (result+(ix_g+3*iy_g+7*iz_g+2))*ia + ic ) % im;
  result = ix_g+3*iy_g+7*iz_g+2;
  result = result & ( (1<<2) - 1 );
#endif
  result = 1 << result;

  return result;
}

/*===========================================================================*/
/*---Scale factor for angles---*/
/*---pseudo-private member function---*/

TARGET_HD static inline int Quantities_scalefactor_angle_( Dimensions dims,
                                                            int ia )
{
  /*---Create a "random" power of 2. Limit the size by taking only
       the low order bits of ia
  ---*/
  Assert( ia >= 0 && ia < dims.na );

  return 1 << ( ia & ( (1<<3) - 1 ) );
}

/*===========================================================================*/
/*---Accessor for weights---*/
/*---pseudo-private member function---*/

TARGET_HD static inline P Quantities_xfluxweight_( Dimensions dims,
                                                    int ia )
{
  Assert( ia >= 0 && ia < dims.na );

  return (P) ( 1 / (P) 2 );
}

/*===========================================================================*/
/*---Accessor for weights---*/
/*---pseudo-private member function---*/

TARGET_HD static inline P Quantities_yfluxweight_( Dimensions dims,
                                                    int ia )
{
  Assert( ia >= 0 && ia < dims.na );

  return (P) ( 1 / (P) 4 );
}

/*===========================================================================*/
/*---Affine function used as basis for vector of linear values.
---pseudo-private member function---*/

TARGET_HD static inline P Quantities_affinefunction_( int i )
{
  /*---NOTE: to insure that the mappings from moments to angles to moments
       restores the original vector, this must be just like this ---*/
  return (P) ( 1 + i );
}

/*===========================================================================*/
/*---Accessor for weights---*/
/*---pseudo-private member function---*/

TARGET_HD static inline P Quantities_zfluxweight_( Dimensions dims,
                                                    int ia )
{
  /*---NOTE: The flux weights are calculated so that each flux "out" is
       identical to any flux "in", under assumptions on the incoming fluxes
       and the state vector at the cell.  The problem boundaries are
       initialized to this required flux value to make this work.
       Note that this value is exactly Quantities_scalefactor_angle_( ia )
       times the state vector value (in angle space) at this cell.
       Powers of 2 are used so that the divides are exact in
       floating point arithmetic.
  ---*/
  Assert( ia >= 0 && ia < dims.na );

  return (P) ( 1 / (P) 4 - 1 / (P) Quantities_scalefactor_angle_( dims, ia ) );
}

/*===========================================================================*/
/*---Scale factor for octants---*/
/*---pseudo-private member function---*/

TARGET_HD static inline int Quantities_scalefactor_octant_( int octant )
{
  Assert( octant>=0 && octant<NOCTANT );

#ifndef RELAXED_TESTING
  const int result = 1 + octant;
#else
  const int result = 1;
#endif

  return result;
}

/*===========================================================================*/
/*---Initial values for boundary array---*/

TARGET_HD static inline P Quantities_init_facexy(
  const Quantities*  quan,
  int                ix_g,
  int                iy_g,
  int                iz_g,
  int                ie,
  int                ia,
  int                iu,
  int                octant,
  const Dimensions   dims_g )
{
  Assert( ix_g >=  0 && ix_g <  dims_g.ncell_x );
  Assert( iy_g >=  0 && iy_g <  dims_g.ncell_y );
  Assert( ( iz_g == -1             && Dir_z(octant)==DIR_UP ) ||
          ( iz_g == dims_g.ncell_z && Dir_z(octant)==DIR_DN ) );
  Assert( ie >=  0 && ie < dims_g.ne );
  Assert( ia >=  0 && ia < dims_g.na );
  Assert( iu >=  0 && iu < NU );
  Assert( octant >= 0 && octant < NOCTANT );

  /*---NOTE: this is constructed to be affine in ia (except for scale factor)
       and independent of ix, iy, iz, to facilitate calculating the
       result analytically---*/

  if( Quantities_bc_vacuum() )
  {
    return ((P)0);
  }
  else
  {
    return   ( (P) Quantities_affinefunction_( ia ) )
           * ( (P) Quantities_scalefactor_angle_( dims_g, ia ) )
           * ( (P) Quantities_scalefactor_space_( quan, ix_g, iy_g, iz_g ) )
           * ( (P) Quantities_scalefactor_energy_( ie, dims_g ) )
           * ( (P) Quantities_scalefactor_unknown_( iu ) )
           * ( (P) Quantities_scalefactor_octant_( octant ) );
  }
}

/*===========================================================================*/
/*---Initial values for boundary array---*/

TARGET_HD static inline P Quantities_init_facexz(
  const Quantities*  quan,
  int                ix_g,
  int                iy_g,
  int                iz_g,
  int                ie,
  int                ia,
  int                iu,
  int                octant,
  const Dimensions   dims_g )
{
  Assert( ix_g >=  0 && ix_g < dims_g.ncell_x );
  Assert( ( iy_g == -1             && Dir_y(octant)==DIR_UP ) ||
          ( iy_g == dims_g.ncell_y && Dir_y(octant)==DIR_DN ) );
  Assert( iz_g >=  0 && iz_g < dims_g.ncell_z );
  Assert( ie >=  0 && ie < dims_g.ne );
  Assert( ia >=  0 && ia < dims_g.na );
  Assert( iu >=  0 && iu < NU );
  Assert( octant >= 0 && octant < NOCTANT );

  if( Quantities_bc_vacuum() )
  {
    return ((P)0);
  }
  else
  {
    return   ( (P) Quantities_affinefunction_( ia ) )
           * ( (P) Quantities_scalefactor_angle_( dims_g, ia ) )
           * ( (P) Quantities_scalefactor_space_( quan, ix_g, iy_g, iz_g ) )
           * ( (P) Quantities_scalefactor_energy_( ie, dims_g ) )
           * ( (P) Quantities_scalefactor_unknown_( iu ) )
           * ( (P) Quantities_scalefactor_octant_( octant ) );
  }
}

/*===========================================================================*/
/*---Initial values for boundary array---*/

TARGET_HD static inline P Quantities_init_faceyz(
  const Quantities*  quan,
  int                ix_g,
  int                iy_g,
  int                iz_g,
  int                ie,
  int                ia,
  int                iu,
  int                octant,
  const Dimensions   dims_g )
{
  Assert( ( ix_g == -1             && Dir_x(octant)==DIR_UP ) ||
          ( ix_g == dims_g.ncell_x && Dir_x(octant)==DIR_DN ) );
  Assert( iy_g >=  0 && iy_g < dims_g.ncell_y );
  Assert( iz_g >=  0 && iz_g < dims_g.ncell_z );
  Assert( ie >=  0 && ie < dims_g.ne );
  Assert( ia >=  0 && ia < dims_g.na );
  Assert( iu >=  0 && iu < NU );
  Assert( octant >= 0 && octant < NOCTANT );

  if( Quantities_bc_vacuum() )
  {
    return ((P)0);
  }
  else
  {
    return   ( (P) Quantities_affinefunction_( ia ) )
           * ( (P) Quantities_scalefactor_angle_( dims_g, ia ) )
           * ( (P) Quantities_scalefactor_space_( quan, ix_g, iy_g, iz_g ) )
           * ( (P) Quantities_scalefactor_energy_( ie, dims_g ) )
           * ( (P) Quantities_scalefactor_unknown_( iu ) )
           * ( (P) Quantities_scalefactor_octant_( octant ) );
  }
}

/*===========================================================================*/
/*---Perform equation solve at a cell---*/

TARGET_HD static inline void Quantities_solve(
  const Quantities* const  quan,
  P* const __restrict__ vslocal,
  const int             ia,
  const int             iaind,
  const int             iamax,
  P* const __restrict__ facexy,
  P* const __restrict__ facexz,
  P* const __restrict__ faceyz,
  const int             ix_b,
  const int             iy_b,
  const int             iz_b,
  const int             ie,
  const int             ix_g,
  const int             iy_g,
  const int             iz_g,
  const int             octant,
  const int             octant_in_block,
  const int             noctant_per_block,
  const Dimensions      dims_b,
  const Dimensions      dims_g,
  const Bool_t          is_cell_active )
{
  Assert( vslocal );
  /*
  Assert( ia >= 0 && ia < dims_b.na );
  */
  Assert( ia >= 0 );
  Assert( iaind >= 0 && iaind < iamax );
  Assert( iamax >= 0 );
  Assert( facexy );
  Assert( facexz );
  Assert( faceyz );
  Assert( ( ix_b >= 0 && ix_b < dims_b.ncell_x ) || ! is_cell_active );
  Assert( ( iy_b >= 0 && iy_b < dims_b.ncell_y ) || ! is_cell_active );
  Assert( ( iz_b >= 0 && iz_b < dims_b.ncell_z ) || ! is_cell_active );
  Assert( ( ix_g >= 0 && ix_g < dims_g.ncell_x ) || ! is_cell_active );
  Assert( ( iy_g >= 0 && iy_g < dims_g.ncell_y ) || ! is_cell_active );
  Assert( ( iz_g >= 0 && iz_g < dims_g.ncell_z ) || ! is_cell_active );
  Assert( ie   >= 0 && ie   < dims_b.ne );
  Assert( octant >= 0 && octant < NOCTANT );
  Assert( octant_in_block >= 0 && octant_in_block < noctant_per_block );

  if( ia < dims_b.na && is_cell_active )
  {
    const int dir_x = Dir_x( octant );
    const int dir_y = Dir_y( octant );
    const int dir_z = Dir_z( octant );

    int iu = 0;

    /*---Average the face values and accumulate---*/

    /*---The state value and incoming face values are first adjusted to
         normalized values by removing the spatial scaling.
         They are then combined using a weighted average chosen in a special
         way to give just the expected result.
         Finally, spatial scaling is applied to the result which is then
         stored.
    ---*/

    const P scalefactor_octant = Quantities_scalefactor_octant_( octant );
    const P scalefactor_octant_r = ((P)1) / scalefactor_octant;
    const P scalefactor_space
                    = Quantities_scalefactor_space_( quan, ix_g, iy_g, iz_g );
    const P scalefactor_space_r = ((P)1) / scalefactor_space;
    const P scalefactor_space_x_r = ((P)1) /
       Quantities_scalefactor_space_( quan, ix_g-Dir_inc(dir_x), iy_g, iz_g );
    const P scalefactor_space_y_r = ((P)1) /
       Quantities_scalefactor_space_( quan, ix_g, iy_g-Dir_inc(dir_y), iz_g );
    const P scalefactor_space_z_r = ((P)1) /
       Quantities_scalefactor_space_( quan, ix_g, iy_g, iz_g-Dir_inc(dir_z) );

#pragma unroll
    for( iu=0; iu<NU; ++iu )
    {
      P* const __restrict__ vslocal_this
                        = ref_vslocal( vslocal, dims_b, NU, iamax, iaind, iu );
/*
      P* const __restrict__ facexy_this
                        = ref_facexy( facexy, dims_b, NU, noctant_per_block,
                                      ix_b, iy_b, ie, ia, iu, octant_in_block );
*/

      const P result = ( *vslocal_this * scalefactor_space_r + (
          *const_ref_facexy( facexy, dims_b, NU, noctant_per_block,
                                     ix_b, iy_b, ie, ia, iu, octant_in_block )
           * Quantities_xfluxweight_( dims_g, ia )
           * scalefactor_space_z_r
        + *const_ref_facexz( facexz, dims_b, NU, noctant_per_block,
                                     ix_b, iz_b, ie, ia, iu, octant_in_block )
           * Quantities_yfluxweight_( dims_g, ia )
           * scalefactor_space_y_r
        + *const_ref_faceyz( faceyz, dims_b, NU, noctant_per_block,
                                     iy_b, iz_b, ie, ia, iu, octant_in_block )
           * Quantities_zfluxweight_( dims_g, ia )
           * scalefactor_space_x_r
      ) * scalefactor_octant_r ) * scalefactor_space;

      *vslocal_this = result;
      const P result_scaled = result * scalefactor_octant;
      *ref_facexy( facexy, dims_b, NU, noctant_per_block,
                   ix_b, iy_b, ie, ia, iu, octant_in_block ) = result_scaled;
      *ref_facexz( facexz, dims_b, NU, noctant_per_block,
                   ix_b, iz_b, ie, ia, iu, octant_in_block ) = result_scaled;
      *ref_faceyz( faceyz, dims_b, NU, noctant_per_block,
                   iy_b, iz_b, ie, ia, iu, octant_in_block ) = result_scaled;
    } /*---for---*/

  }
} /*---Quantities_solve---*/

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_quantities_testing_kernels_h_---*/

/*---------------------------------------------------------------------------*/
