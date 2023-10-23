/*---------------------------------------------------------------------------*/
/*!
 * \file   sweeper_tileoctants_c.h
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Definitions for performing a sweep, tileoctants version.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _sweeper_tileoctants_c_h_
#define _sweeper_tileoctants_c_h_

#include "env.h"
#include "definitions.h"
#include "quantities.h"
#include "array_accessors.h"
#include "array_operations.h"
#include "sweeper_tileoctants.h"

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
  const Bool_t do_tile_octants = Sweeper_tile_octants();
  const int ntilestep = do_tile_octants ? NOCTANT : 1;
  const Dimensions dims = sweeper->dims;

  int tilestep = 0;

  /*---Initialize result array to zero---*/

  initialize_state_zero( Pointer_h( vo ), dims, NU );

  /*---Loop over octant tiles---*/

  /*---If tiling is requested, at each of the 8 tile steps,
       each octant direction computes its result on a
       1/8-sized tile of the domain.
       This is scheduled such that: 1) the proper sweep order for each
       octant direction is adhered to, and 2) for each tile step, the
       8 octant directions are working on independent disjoint
       tiles of the domain.
       NOTE: if the octants are OpenMP-threaded, then the loop order
       must be this way (tile then octant), and there should be a sync
       between consecutive tile steps.
  ---*/

  for( tilestep=0; tilestep<ntilestep; ++tilestep )
  {

  /*---Loop over octants---*/

  for( octant=0; octant<NOCTANT; ++octant )
  {
    /*---If tiling, then each octant direction needs its own face:
         each octant direction is not processed all-at-once but
         intermittently, thus a need to remember its state.
    ---*/

    const int octant_in_block = do_tile_octants ? octant : 0;
    Assert( octant_in_block >= 0 &&
            octant_in_block < Sweeper_noctant_per_block( sweeper ) );

    /*---Decode octant directions from octant number---*/

    const int dir_x = Dir_x( octant );
    const int dir_y = Dir_y( octant );
    const int dir_z = Dir_z( octant );

    /*---Determine tile to be computed---*/

    /*---The scheduling works right if each octant direction visits tiles
         in lexicographical ordering x-varies-fastest, starting at
         the origin corner of the octant direction.
         Note the tiles are defined by lower or upper half of domain
         in each of x, y and z.

         NOTE: this strategy is tested for the single thread case
         but not yet tested under OpenMP.
    ---*/

    const int tile_x = (!do_tile_octants) ? 0 :
           ( ( tilestep & (1<<0) ) == 0 ) == ( dir_x == DIR_UP ) ? DIR_LO
                                                                 : DIR_HI;
    const int tile_y = (!do_tile_octants) ? 0 :
           ( ( tilestep & (1<<1) ) == 0 ) == ( dir_y == DIR_UP ) ? DIR_LO
                                                                 : DIR_HI;
    const int tile_z = (!do_tile_octants) ? 0 :
           ( ( tilestep & (1<<2) ) == 0 ) == ( dir_z == DIR_UP ) ? DIR_LO
                                                                 : DIR_HI;

    /*---Compute tile boundaries---*/

    /*---If no tiling, then whole domain, otherwise 1/2 of
         domain in each direction
    ---*/

    const int tile_xmin = (!do_tile_octants) ? 0         :
                          tile_x==DIR_LO     ? 0         : dims.ncell_x/2;
    const int tile_ymin = (!do_tile_octants) ? 0         :
                          tile_y==DIR_LO     ? 0         : dims.ncell_y/2;
    const int tile_zmin = (!do_tile_octants) ? 0         :
                          tile_z==DIR_LO     ? 0         : dims.ncell_z/2;

    const int tile_xmax = (!do_tile_octants) ? dims.ncell_x   :
                          tile_x==DIR_LO     ? dims.ncell_x/2 : dims.ncell_x;
    const int tile_ymax = (!do_tile_octants) ? dims.ncell_y   :
                          tile_y==DIR_LO     ? dims.ncell_y/2 : dims.ncell_y;
    const int tile_zmax = (!do_tile_octants) ? dims.ncell_z   :
                          tile_z==DIR_LO     ? dims.ncell_z/2 : dims.ncell_z;

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
         Note if no tiling then we initialize all faces here,
         otherwise only the appropriate part on the appropriate
         tiling step.
    ---*/

    if( tile_z != dir_z || !do_tile_octants )
    {
      iz = dir_z==DIR_UP ? -1 : dims.ncell_z;
      for( iu=0; iu<NU; ++iu )
      for( iy=tile_ymin; iy<tile_ymax; ++iy )
      for( ix=tile_xmin; ix<tile_xmax; ++ix )
      for( ie=0; ie<dims.ne; ++ie )
      for( ia=0; ia<dims.na; ++ia )
      {
        *ref_facexy( sweeper->facexy, dims, NU,
                     Sweeper_noctant_per_block( sweeper ),
                     ix, iy, ie, ia, iu, octant_in_block )
             = Quantities_init_facexy(
                                  quan, ix, iy, iz, ie, ia, iu, octant, dims );
      }
    }

    if( tile_y != dir_y || !do_tile_octants )
    {
      iy = dir_y==DIR_UP ? -1 : dims.ncell_y;
      for( iu=0; iu<NU; ++iu )
      for( iz=tile_zmin; iz<tile_zmax; ++iz )
      for( ix=tile_xmin; ix<tile_xmax; ++ix )
      for( ie=0; ie<dims.ne; ++ie )
      for( ia=0; ia<dims.na; ++ia )
      {
        *ref_facexz( sweeper->facexz, dims, NU,
                     Sweeper_noctant_per_block( sweeper ),
                     ix, iz, ie, ia, iu, octant_in_block )
             = Quantities_init_facexz(
                                  quan, ix, iy, iz, ie, ia, iu, octant, dims );
      }
    }

    if( tile_x != dir_x || !do_tile_octants )
    {
      ix = dir_x==DIR_UP ? -1 : dims.ncell_x;
      for( iu=0; iu<NU; ++iu )
      for( iz=tile_zmin; iz<tile_zmax; ++iz )
      for( iy=tile_ymin; iy<tile_ymax; ++iy )
      for( ie=0; ie<dims.ne; ++ie )
      for( ia=0; ia<dims.na; ++ia )
      {
        *ref_faceyz( sweeper->faceyz, dims, NU,
                     Sweeper_noctant_per_block( sweeper ),
                     iy, iz, ie, ia, iu, octant_in_block )
             = Quantities_init_faceyz(
                                  quan, ix, iy, iz, ie, ia, iu, octant, dims );
      }
    }

    /*---Loop over energy groups---*/

    for( ie=0; ie<dims.ne; ++ie )
    {
      /*---Calculate spatial loop extents, possibly based on tiling---*/

      const int ixbeg = dir_x==DIR_UP ? tile_xmin : tile_xmax-1;
      const int iybeg = dir_y==DIR_UP ? tile_ymin : tile_ymax-1;
      const int izbeg = dir_z==DIR_UP ? tile_zmin : tile_zmax-1;

      const int ixend = dir_x==DIR_DN ? tile_xmin : tile_xmax-1;
      const int iyend = dir_y==DIR_DN ? tile_ymin : tile_ymax-1;
      const int izend = dir_z==DIR_DN ? tile_zmin : tile_zmax-1;

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
      for( ia=0; ia<dims.na; ++ia )
      {
        P result = P_zero();
        for( im=0; im<dims.nm; ++im )
        {
          result += *const_ref_a_from_m( Pointer_const_h( & quan->a_from_m ),
                                         dims, im, ia, octant )*
                    *const_ref_state( Pointer_h( vi ), dims, NU, ix, iy, iz, ie, im, iu );
        }
        *ref_vslocal( sweeper->vslocal, dims, NU, dims.na, ia, iu ) = result;
      }

      /*--------------------*/
      /*---Perform solve---*/
      /*--------------------*/

      for( ia=0; ia<dims.na; ++ia )
      {
        Quantities_solve( quan, sweeper->vslocal, ia, ia, dims.na,
                          sweeper->facexy, sweeper->facexz, sweeper->faceyz,
                          ix, iy, iz, ie, ix, iy, iz, 
                          octant, octant_in_block,
                          Sweeper_noctant_per_block( sweeper ),
                          dims, dims, Bool_true );
      }

      /*--------------------*/
      /*---Transform state vector from angles to moments---*/
      /*--------------------*/

      /*---Perform small dense matrix-vector products and store
           the result in the output state vector.
      ---*/

      for( iu=0; iu<NU; ++iu )
      for( im=0; im<dims.nm; ++im )
      {
        P result = P_zero();
        for( ia=0; ia<dims.na; ++ia )
        {
          result += *const_ref_m_from_a( Pointer_const_h( & quan->m_from_a ),
                                         dims, im, ia, octant )*
                    *const_ref_vslocal( sweeper->vslocal, dims, NU, dims.na, ia, iu );
        }
        *ref_state( Pointer_h( vo ), dims, NU, ix, iy, iz, ie, im, iu ) += result;
      }

    } /*---ix/iy/iz---*/

    } /*---ie---*/

  } /*---octant---*/

  } /*---octant_tile---*/

} /*---sweep---*/

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_sweeper_tileoctants_c_h_---*/

/*---------------------------------------------------------------------------*/
