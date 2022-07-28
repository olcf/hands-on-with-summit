/*---------------------------------------------------------------------------*/
/*!
 * \file   sweeper_kba.h
 * \author Wayne Joubert
 * \date   Tue Jan 28 16:37:41 EST 2014
 * \brief  Declarations for performing a sweep, kba version.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _sweeper_kba_h_
#define _sweeper_kba_h_

#include "types.h"
#include "env.h"
#include "definitions.h"
#include "dimensions.h"
#include "arguments.h"
#include "pointer.h"
#include "quantities.h"
#include "stepscheduler_kba.h"
#include "faces_kba.h"

#include "sweeper_kba_kernels.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Struct with pointers etc. used to perform sweep---*/

typedef struct
{
  P* __restrict__  vilocal_host_;
  P* __restrict__  vslocal_host_;
  P* __restrict__  volocal_host_;

  Dimensions       dims;
  Dimensions       dims_b;
  Dimensions       dims_g;

  int              nthread_e;
  int              nthread_octant;
  int              nthread_x;
  int              nthread_y;
  int              nthread_z;

  int              nblock_z;
  int              nblock_octant;
  int              noctant_per_block;
  int              nsemiblock;
  int              ncell_x_per_subblock;
  int              ncell_y_per_subblock;
  int              ncell_z_per_subblock;

  StepScheduler    stepscheduler;

  Faces            faces;
} Sweeper;

/*===========================================================================*/
/*---Null object---*/

Sweeper Sweeper_null(void);

/*===========================================================================*/
/*---Pseudo-constructor for Sweeper struct---*/

void Sweeper_create( Sweeper*          sweeper,
                     Dimensions        dims,
                     const Quantities* quan,
                     Env*              env,
                     Arguments*        args );

/*===========================================================================*/
/*---Pseudo-destructor for Sweeper struct---*/

void Sweeper_destroy( Sweeper* sweeper,
                      Env*     env );

/*===========================================================================*/
/*---Number of octants in an octant block---*/

static int Sweeper_noctant_per_block( const Sweeper* sweeper )
{
  return sweeper->noctant_per_block;
}

/*===========================================================================*/
/*---Thread counts for amu for execution target as understood by the host---*/

static inline int Sweeper_nthread_a( const Sweeper* sweeper,
                                     const Env*     env )
{
  return Env_cuda_is_using_device( env ) ? NTHREAD_DEVICE_A*1 : NTHREAD_A*1;
}

/*---------------------------------------------------------------------------*/

static inline int Sweeper_nthread_m( const Sweeper* sweeper,
                                     const Env*     env )
{
  return Env_cuda_is_using_device( env ) ? NTHREAD_DEVICE_M*1 : NTHREAD_M*1;
}

/*---------------------------------------------------------------------------*/

static inline int Sweeper_nthread_u( const Sweeper* sweeper,
                                     const Env*     env )
{
  return Env_cuda_is_using_device( env ) ? NTHREAD_DEVICE_U*1 : NTHREAD_U*1;
}

/*===========================================================================*/
/*---Number of elements to allocate for v*local---*/

static inline int Sweeper_nvilocal_( Sweeper* sweeper,
                                      Env*     env )
{
  return Env_cuda_is_using_device( env )
      ?
         Sweeper_nthread_m( sweeper, env ) *
         NU *
         sweeper->nthread_octant *
         sweeper->nthread_x *
         sweeper->nthread_y *
         sweeper->nthread_z
       :
         Sweeper_nthread_m( sweeper, env ) *
         NU *
         sweeper->nthread_octant *
         sweeper->nthread_e *
         sweeper->nthread_x *
         sweeper->nthread_y *
         sweeper->nthread_z
       ;
}

/*---------------------------------------------------------------------------*/

static inline int Sweeper_nvslocal_( Sweeper* sweeper,
                                      Env*     env )
{
  return Env_cuda_is_using_device( env )
      ?
         Sweeper_nthread_a( sweeper, env ) *
         NU *
         sweeper->nthread_octant *
         sweeper->nthread_x *
         sweeper->nthread_y *
         sweeper->nthread_z
       :
         Sweeper_nthread_a( sweeper, env ) *
         NU *
         sweeper->nthread_octant *
         sweeper->nthread_e *
         sweeper->nthread_x *
         sweeper->nthread_y *
         sweeper->nthread_z
       ;
}

/*---------------------------------------------------------------------------*/

static inline int Sweeper_nvolocal_( Sweeper* sweeper,
                                      Env*     env )
{
  return Env_cuda_is_using_device( env )
      ?
         Sweeper_nthread_m( sweeper, env ) *
         NU *
         sweeper->nthread_octant *
         sweeper->nthread_x *
         sweeper->nthread_y *
         sweeper->nthread_z
       :
         Sweeper_nthread_m( sweeper, env ) *
         NU *
         sweeper->nthread_octant *
         sweeper->nthread_e *
         sweeper->nthread_x *
         sweeper->nthread_y *
         sweeper->nthread_z
       ;
}

/*===========================================================================*/
/*---For kernel launch: CUDA thread/block counts---*/

static int Sweeper_nthreadblock( const Sweeper* sweeper,
                                 int            axis,
                                 Env*           env )
{
  Assert( axis >= 0 && axis < 3 );

  return axis==0 ? sweeper->nthread_e :
                   1;
}

/*---------------------------------------------------------------------------*/

static int Sweeper_nthread_in_threadblock( const Sweeper* sweeper,
                                           int            axis,
                                           Env*           env )
{
  Assert( axis >= 0 && axis < 3 );

  return axis==0 ? Sweeper_nthread_a( sweeper, env ) :
         axis==1 ? sweeper->nthread_octant :
                   sweeper->nthread_y *
                   sweeper->nthread_z;
}

/*===========================================================================*/
/*---Fof kernel launch: full size of CUDA shared memory---*/

static int Sweeper_shared_size_( Sweeper* sweeper,
                                 Env*     env )
{
  return ( Sweeper_nvilocal_( sweeper, env ) +
           Sweeper_nvslocal_( sweeper, env ) +
           Sweeper_nvolocal_( sweeper, env ) ) * sizeof( P );
}

/*===========================================================================*/
/*---Extract SweeperLite from Sweeper---*/

SweeperLite Sweeper_sweeperlite( Sweeper* sweeper );

/*===========================================================================*/
/*---Perform a sweep for a block---*/

void Sweeper_sweep_block(
  Sweeper*               sweeper,
  Pointer*               vo,
  Pointer*               vi,
  int*                   is_block_init,
  Pointer*               facexy,
  Pointer*               facexz,
  Pointer*               faceyz,
  const Pointer*         a_from_m,
  const Pointer*         m_from_a,
  int                    step,
  const Quantities*      quan,
  Env*                   env );

/*===========================================================================*/
/*---Perform a sweep---*/

void Sweeper_sweep(
  Sweeper*               sweeper,
  Pointer*               vo,
  Pointer*               vi,
  const Quantities*      quan,
  Env*                   env );

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_sweeper_kba_h_---*/

/*---------------------------------------------------------------------------*/
