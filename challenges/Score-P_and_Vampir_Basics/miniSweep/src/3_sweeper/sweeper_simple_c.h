/*---------------------------------------------------------------------------*/
/*!
 * \file   sweeper_simple_c.h
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Definitions for performing a sweep, simple version.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _sweeper_simple_c_h_
#define _sweeper_simple_c_h_

#include "env.h"
#include "definitions.h"
#include "quantities.h"
#include "array_accessors.h"
#include "array_operations.h"
#include "sweeper_simple.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Null object---*/

Sweeper Sweeper_null()
{
  Sweeper result;
  memset( (void*)&result, 0, sizeof(Sweeper) );
  return result;
}

/*===========================================================================*/
/*---Pseudo-constructor for Sweeper struct---*/

void Sweeper_create( Sweeper*          sweeper,
                     Dimensions        dims,
                     const Quantities* quan,
                     Env*              env,
                     Arguments*        args )
{
  Insist( Env_nproc( env ) == 1 && 
                             "This sweeper version runs only with one proc." );

  /*---Allocate arrays---*/

  sweeper->vslocal = malloc_host_P( dims.na * NU );
  sweeper->facexy  = malloc_host_P( dims.ncell_x * dims.ncell_y * dims.ne *
                         dims.na * NU * Sweeper_noctant_per_block( sweeper ) );
  sweeper->facexz  = malloc_host_P( dims.ncell_x * dims.ncell_z * dims.ne *
                         dims.na * NU * Sweeper_noctant_per_block( sweeper ) );
  sweeper->faceyz  = malloc_host_P( dims.ncell_y * dims.ncell_z * dims.ne *
                         dims.na * NU * Sweeper_noctant_per_block( sweeper ) );

  sweeper->dims = dims;
}

/*===========================================================================*/
/*---Pseudo-destructor for Sweeper struct---*/

void Sweeper_destroy( Sweeper* sweeper,
                      Env*     env )
{
  /*---Deallocate arrays---*/

  free_host_P( sweeper->vslocal );
  free_host_P( sweeper->facexy );
  free_host_P( sweeper->facexz );
  free_host_P( sweeper->faceyz );

  sweeper->vslocal = NULL;
  sweeper->facexy  = NULL;
  sweeper->facexz  = NULL;
  sweeper->faceyz  = NULL;
}

/*===========================================================================*/
/*---Perform a sweep---*/

void Sweeper_sweep(
  Sweeper*               sweeper,
  Pointer*               vo,
  Pointer*               vi,
  const Quantities*      quan,
  Env*                   env )
{
  Assert( sweeper );
  Assert( vi );
  Assert( vo );

  /*---Declarations---*/
  int ix = 0;
  int iy = 0;
  int iz = 0;
  int ie = 0;
  int im = 0;
  int ia = 0;
  int iu = 0;
  int octant = 0;

  /*---Initialize result array to zero---*/

  initialize_state_zero( Pointer_h( vo ), sweeper->dims, NU );

  /*---Loop over octants---*/

  for( octant=0; octant<NOCTANT; ++octant )
  {
    const int octant_in_block = 0;
    Assert( octant_in_block >= 0 &&
            octant_in_block < Sweeper_noctant_per_block( sweeper ) );

    /*---Decode octant directions from octant number---*/

    const int dir_x = Dir_x( octant );
    const int dir_y = Dir_y( octant );
    const int dir_z = Dir_z( octant );

    /*---Initialize faces---*/

    /*---The semantics of the face arrays are as follows.
         On entering a cell for a solve at the gridcell level,
         the face array is assumed to have a value corresponding to
         "one cell lower" in the relevant direction.
         On leaving the gridcell solve, the face has been updated
         to have the flux at that gridcell.
         Thus, the face is initialized at first to have a value
         "one cell" outside of the domain, e.g., for the XY face,
         either -1 or dims.ncell_x.
         Note also that the face initializer functions now take
         coordinates for all three spatial dimensions --
         the third dimension is used to denote whether it is the
         "lower" or "upper" face and also its exact location
         in that dimension.
    ---*/

    {
      iz = dir_z == DIR_UP ? -1 : sweeper->dims.ncell_z;
      for( iu=0; iu<NU; ++iu )
      for( iy=0; iy<sweeper->dims.ncell_y; ++iy )
      for( ix=0; ix<sweeper->dims.ncell_x; ++ix )
      for( ie=0; ie<sweeper->dims.ne; ++ie )
      for( ia=0; ia<sweeper->dims.na; ++ia )
      {
        *ref_facexy( sweeper->facexy, sweeper->dims, NU,
                     Sweeper_noctant_per_block( sweeper ),
                     ix, iy, ie, ia, iu, octant_in_block )
             = Quantities_init_facexy(
                         quan, ix, iy, iz, ie, ia, iu, octant, sweeper->dims );
      }
    }

    {
      iy = dir_y == DIR_UP ? -1 : sweeper->dims.ncell_y;
      for( iu=0; iu<NU; ++iu )
      for( iz=0; iz<sweeper->dims.ncell_z; ++iz )
      for( ix=0; ix<sweeper->dims.ncell_x; ++ix )
      for( ie=0; ie<sweeper->dims.ne; ++ie )
      for( ia=0; ia<sweeper->dims.na; ++ia )
      {
        *ref_facexz( sweeper->facexz, sweeper->dims, NU,
                     Sweeper_noctant_per_block( sweeper ),
                     ix, iz, ie, ia, iu, octant_in_block )
             = Quantities_init_facexz(
                         quan, ix, iy, iz, ie, ia, iu, octant, sweeper->dims );
      }
    }

    {
      ix = dir_x == DIR_UP ? -1 : sweeper->dims.ncell_x;
      for( iu=0; iu<NU; ++iu )
      for( iz=0; iz<sweeper->dims.ncell_z; ++iz )
      for( iy=0; iy<sweeper->dims.ncell_y; ++iy )
      for( ie=0; ie<sweeper->dims.ne; ++ie )
      for( ia=0; ia<sweeper->dims.na; ++ia )
      {
        *ref_faceyz( sweeper->faceyz, sweeper->dims, NU,
                     Sweeper_noctant_per_block( sweeper ),
                     iy, iz, ie, ia, iu, octant_in_block )
             = Quantities_init_faceyz(
                         quan, ix, iy, iz, ie, ia, iu, octant, sweeper->dims );
      }
    }

    /*---Loop over energy groups---*/

    for( ie=0; ie<sweeper->dims.ne; ++ie )
    {
      /*---Calculate spatial loop extents---*/

      int ixbeg = dir_x==DIR_UP ? 0 : sweeper->dims.ncell_x-1;
      int iybeg = dir_y==DIR_UP ? 0 : sweeper->dims.ncell_y-1;
      int izbeg = dir_z==DIR_UP ? 0 : sweeper->dims.ncell_z-1;

      int ixend = dir_x==DIR_DN ? 0 : sweeper->dims.ncell_x-1;
      int iyend = dir_y==DIR_DN ? 0 : sweeper->dims.ncell_y-1;
      int izend = dir_z==DIR_DN ? 0 : sweeper->dims.ncell_z-1;

      /*---Loop over cells, in proper direction---*/

    for( iz=izbeg; iz!=izend+Dir_inc(dir_z); iz+=Dir_inc(dir_z) )
    for( iy=iybeg; iy!=iyend+Dir_inc(dir_y); iy+=Dir_inc(dir_y) )
    for( ix=ixbeg; ix!=ixend+Dir_inc(dir_x); ix+=Dir_inc(dir_x) )
    {

      /*--------------------*/
      /*---Transform state vector from moments to angles---*/
      /*--------------------*/

      /*---This loads values from the input state vector,
           does the small dense matrix-vector product,
           and stores the result in a relatively small local
           array that is hopefully small enough to fit into
           processor cache.
      ---*/

      for( iu=0; iu<NU; ++iu )
      for( ia=0; ia<sweeper->dims.na; ++ia )
      {
        P result = P_zero();
        for( im=0; im<sweeper->dims.nm; ++im )
        {
          result += *const_ref_a_from_m( Pointer_const_h( & quan->a_from_m ),
                                         sweeper->dims, im, ia, octant )*
                    *const_ref_state(    Pointer_h( vi ), sweeper->dims, NU,
                                         ix, iy, iz, ie, im, iu );
        }
        *ref_vslocal( sweeper->vslocal, sweeper->dims, NU,
                      sweeper->dims.na, ia, iu ) = result;
      }

      /*--------------------*/
      /*---Perform solve---*/
      /*--------------------*/

      for( ia=0; ia<sweeper->dims.na; ++ia )
      {
        Quantities_solve( quan, sweeper->vslocal, ia, ia, sweeper->dims.na,
                          sweeper->facexy, sweeper->facexz, sweeper->faceyz,
                          ix, iy, iz, ie, ix, iy, iz,
                          octant, octant_in_block,
                          Sweeper_noctant_per_block( sweeper ),
                          sweeper->dims, sweeper->dims, Bool_true );
      }

      /*--------------------*/
      /*---Transform state vector from angles to moments---*/
      /*--------------------*/

      /*---Perform small dense matrix-vector products and store
           the result in the output state vector.
      ---*/

      for( iu=0; iu<NU; ++iu )
      for( im=0; im<sweeper->dims.nm; ++im )
      {
        P result = P_zero();
        for( ia=0; ia<sweeper->dims.na; ++ia )
        {
          result += *const_ref_m_from_a( Pointer_const_h( & quan->m_from_a ),
                                         sweeper->dims, im, ia, octant )*
                    *const_ref_vslocal(  sweeper->vslocal, sweeper->dims, NU,
                                         sweeper->dims.na, ia, iu );
        }
        *ref_state( Pointer_h( vo ), sweeper->dims, NU,
                    ix, iy, iz, ie, im, iu ) += result;
      }

    } /*---ix/iy/iz---*/

    } /*---ie---*/

  } /*---octant---*/

} /*---sweep---*/

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_sweeper_simple_c_h_---*/

/*---------------------------------------------------------------------------*/
