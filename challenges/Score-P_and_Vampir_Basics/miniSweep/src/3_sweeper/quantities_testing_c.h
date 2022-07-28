/*---------------------------------------------------------------------------*/
/*!
 * \file   quantities_testing_c.h
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Definitions for physical quantities, testing case.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _quantities_testing_c_h_
#define _quantities_testing_c_h_

#include "env.h"
#include "dimensions.h"
#include "array_accessors.h"
#include "pointer.h"
#include "quantities_testing.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Pseudo-constructor for Quantities struct---*/

void Quantities_create( Quantities*       quan,
                        const Dimensions  dims,
                        Env*              env )
{
  Quantities_init_am_matrices_( quan, dims, env );
  Quantities_init_decomp_( quan, dims, env );

} /*---Quantities_create---*/

/*===========================================================================*/
/*---Initialize Quantities a_from_m, m_from_a matrices---*/

void Quantities_init_am_matrices_( Quantities*       quan,
                                   const Dimensions  dims,
                                   Env*              env )
{
  /*---Declarations---*/

  int im     = 0;
  int ia     = 0;
  int i      = 0;
  int octant = 0;

  /*---Allocate arrays---*/

  Pointer_create( & quan->a_from_m, dims.nm * dims.na * NOCTANT,
                                             Env_cuda_is_using_device( env ) );
  Pointer_create( & quan->m_from_a, dims.nm * dims.na * NOCTANT,
                                             Env_cuda_is_using_device( env ) );

  Pointer_allocate( & quan->a_from_m );
  Pointer_allocate( & quan->m_from_a );

  /*-----------------------------*/
  /*---Set entries of a_from_m---*/
  /*-----------------------------*/

  /*---These two matrices are set in a special way so as to map a vector
       whose values satisfy a linear relationship v_i = a * i + b
       are mapped to another vector with same property.  This is to facilitate
       being able to have an analytical solution for the sweep.
  ---*/

  /*---First set to zero---*/

  for( octant=0; octant<NOCTANT; ++octant )
  for( im=0;     im<dims.nm;     ++im )
  for( ia=0;     ia<dims.na;     ++ia )
  {
    *ref_a_from_m( Pointer_h( & quan->a_from_m ), dims, im, ia, octant )
                                                                   = P_zero();
  }

  /*---Map a linear vector of values to a similar linear vector of values---*/

  /*---The input state vector in the moment dimension is contrived to
       satisfy vi[im] = 1 + im, an affine function, possibly times a constant.
       The following matrix is artifically contrived to send this to
       a result satisfying, in angle, vl[ia] = 1 + ia, possibly times a
       constant.
  ---*/

  for( octant=0; octant<NOCTANT; ++octant )
  for( i=0;      i<dims.na;      ++i )
  {
    const int quot = ( i + 1 ) / dims.nm;
    const int rem  = ( i + 1 ) % dims.nm;

    *ref_a_from_m( Pointer_h( & quan->a_from_m ), dims, dims.nm-1, i, octant )
                                                                       += quot;
    if( rem != 0 )
    {
      *ref_a_from_m( Pointer_h( & quan->a_from_m ), dims, 0,   i, octant )
                                                                   += -P_one();
      *ref_a_from_m( Pointer_h( & quan->a_from_m ), dims, rem, i, octant )
                                                                   +=  P_one();
    }
  }

  /*---Fill matrix with entries that leave linears unaffected---*/

  /*---This is to create a more dense, nontrivial matrix, with additions
       to the rows that are guaranteed to send affine functions to zero.
  ---*/

  for( octant=0; octant<NOCTANT; ++octant )
  for( im=0;     im<dims.nm-2;   ++im )
  for( ia=0;     ia<dims.na;     ++ia )
  {
    const int randvalue = 21 + ( im + dims.nm * ia ) % 17;

    *ref_a_from_m( Pointer_h( & quan->a_from_m ), dims, im+0, ia, octant ) +=
                                                          -P_one() * randvalue;
    *ref_a_from_m( Pointer_h( & quan->a_from_m ), dims, im+1, ia, octant ) +=
                                                         2*P_one() * randvalue;
    *ref_a_from_m( Pointer_h( & quan->a_from_m ), dims, im+2, ia, octant ) +=
                                                          -P_one() * randvalue;
  }

  /*-----------------------------*/
  /*---Set entries of m_from_a---*/
  /*-----------------------------*/

  /*---First set to zero---*/

  for( octant=0; octant<NOCTANT; ++octant )
  for( im=0;     im<dims.nm;     ++im )
  for( ia=0;     ia<dims.na;     ++ia )
  {
    *ref_m_from_a( Pointer_h( & quan->m_from_a ), dims, im, ia, octant )
                                                                    = P_zero();
  }

  /*---Map a linear vector of values to a similar linear vector of values---*/

  /*---As previously, send functions vl[ia] = 1 + ia to functions
       vo[im] = 1 + im, subject to possible constant scalings
       and also to a power-of-two angle scalefactor adjustment
       designed to make the test more rigorous.
  ---*/

  for( octant=0; octant<NOCTANT; ++octant )
  for( i=0;      i<dims.nm;      ++i )
  {
    const int quot = ( i + 1 ) / dims.na;
    const int rem  = ( i + 1 ) % dims.na;

    *ref_m_from_a( Pointer_h( & quan->m_from_a ), dims, i, dims.na-1, octant )
                                                                       += quot;
    if( rem != 0 )
    {
      *ref_m_from_a( Pointer_h( & quan->m_from_a ), dims, i, 0  , octant )
                                                                   += -P_one();
      *ref_m_from_a( Pointer_h( & quan->m_from_a ), dims, i, rem, octant )
                                                                   +=  P_one();
    }
  }

  /*---Fill matrix with entries that leave linears unaffected---*/

  /*---As before, create more complicated matrix by adding to rows
       entries that do not affect the scaled-affine input values expected.
  ---*/

  for( octant=0; octant<NOCTANT; ++octant )
  for( im=0;     im<dims.nm;     ++im )
  for( ia=0;     ia<dims.na-2;   ++ia )
  {
    const int randvalue = 37 + ( im + dims.nm * ia ) % 19;

    *ref_m_from_a( Pointer_h( & quan->m_from_a ), dims, im, ia+0, octant ) +=
                                                          -P_one() * randvalue;
    *ref_m_from_a( Pointer_h( & quan->m_from_a ), dims, im, ia+1, octant ) +=
                                                         2*P_one() * randvalue;
    *ref_m_from_a( Pointer_h( & quan->m_from_a ), dims, im, ia+2, octant ) +=
                                                          -P_one() * randvalue;
  }

  /*---Scale matrix to compensate for 8 octants and also angle scale factor---*/

  for( octant=0; octant<NOCTANT; ++octant )
  for( im=0;     im<dims.nm;     ++im )
  for( ia=0;     ia<dims.na;     ++ia )
  {
    *ref_m_from_a( Pointer_h( & quan->m_from_a ), dims, im, ia, octant )
                                                                    /= NOCTANT;
    *ref_m_from_a( Pointer_h( & quan->m_from_a ), dims, im, ia, octant )
                                 /= Quantities_scalefactor_angle_( dims, ia );
  }

  Pointer_update_d( & quan->a_from_m );
  Pointer_update_d( & quan->m_from_a );

} /*---Quantities_init_am_matrices_---*/

/*===========================================================================*/
/*---Initialize Quantities subgrid decomp info---*/

void Quantities_init_decomp_( Quantities*       quan,
                              const Dimensions  dims,
                              Env*              env )
{
  /*---Declarations---*/

  int i  = 0;

  /*---Record z dimension---*/

  quan->ncell_z_g = dims.ncell_z;

  /*---Allocate arrays---*/

  quan->ix_base_vals = malloc_host_int( Env_nproc_x( env ) + 1 );
  quan->iy_base_vals = malloc_host_int( Env_nproc_y( env ) + 1 );

  /*---------------------------------*/
  /*---Set entries of ix_base_vals---*/
  /*---------------------------------*/

  /*---Collect values to base proc along axis---*/

  if( Env_proc_x_this( env ) == 0 )
  {
    int proc_x = 0;
    quan->ix_base_vals[ 1+0 ] = dims.ncell_x;
    for( proc_x=1; proc_x<Env_nproc_x( env ); ++proc_x )
    {
      Env_recv_i( env, & quan->ix_base_vals[ 1+proc_x ], 1,
        Env_proc( env, proc_x, Env_proc_y_this( env ) ), Env_tag( env ) );
    }
  }
  else
  {
    Env_send_i( env, & dims.ncell_x, 1,
             Env_proc( env, 0, Env_proc_y_this( env ) ), Env_tag( env ) );
  }
  Env_increment_tag( env, 1 );

  /*---Broadcast collected array to all other procs along axis---*/

  if( Env_proc_x_this( env ) == 0 )
  {
    int proc_x = 0;
    for( proc_x=1; proc_x<Env_nproc_x( env ); ++proc_x )
    {
      Env_send_i( env, & quan->ix_base_vals[ 1 ], Env_nproc_x( env ),
        Env_proc( env, proc_x, Env_proc_y_this( env ) ), Env_tag( env ) );
    }
  }
  else
  {
    Env_recv_i( env, & quan->ix_base_vals[ 1 ], Env_nproc_x( env ),
             Env_proc( env, 0, Env_proc_y_this( env ) ), Env_tag( env ) );
  }
  Env_increment_tag( env, 1 );

  /*---Scan sum---*/

  quan->ix_base_vals[0] = 0;
  for( i=0; i<Env_nproc_x( env ); ++i )
  {
    quan->ix_base_vals[1+i] += quan->ix_base_vals[i];
  }

  quan->ix_base   = quan->ix_base_vals[ Env_proc_x_this( env ) ];
  quan->ncell_x_g = quan->ix_base_vals[ Env_nproc_x(     env ) ];

  Assert( quan->ix_base_vals[ Env_proc_x_this( env )+1 ] -
          quan->ix_base_vals[ Env_proc_x_this( env )   ] == dims.ncell_x );

  /*---------------------------------*/
  /*---Set entries of iy_base_vals---*/
  /*---------------------------------*/

  /*---Collect values to base proc along axis---*/

  if( Env_proc_y_this( env ) == 0 )
  {
    int proc_y = 0;
    quan->iy_base_vals[ 1+0 ] = dims.ncell_y;
    for( proc_y=1; proc_y<Env_nproc_y( env ); ++proc_y )
    {
      Env_recv_i( env, & quan->iy_base_vals[ 1+proc_y ], 1,
        Env_proc( env, Env_proc_x_this( env ), proc_y ), Env_tag( env ) );
    }
  }
  else
  {
    Env_send_i( env, & dims.ncell_y, 1,
             Env_proc( env, Env_proc_x_this( env ), 0 ), Env_tag( env ) );
  }
  Env_increment_tag( env, 1 );

  /*---Broadcast collected array to all other procs along axis---*/

  if( Env_proc_y_this( env ) == 0 )
  {
    int proc_y = 0;
    for( proc_y=1; proc_y<Env_nproc_y( env ); ++proc_y )
    {
      Env_send_i( env, & quan->iy_base_vals[ 1 ], Env_nproc_y( env ),
        Env_proc( env, Env_proc_x_this( env ), proc_y ), Env_tag( env ) );
    }
  }
  else
  {
    Env_recv_i( env, & quan->iy_base_vals[ 1 ], Env_nproc_y( env ),
             Env_proc( env, Env_proc_x_this( env ), 0 ), Env_tag( env ) );
  }
  Env_increment_tag( env, 1 );

  /*---Scan sum---*/

  quan->iy_base_vals[0] = 0;
  for( i=0; i<Env_nproc_y( env ); ++i )
  {
    quan->iy_base_vals[1+i] += quan->iy_base_vals[i];
  }

  quan->iy_base   = quan->iy_base_vals[ Env_proc_y_this( env ) ];
  quan->ncell_y_g = quan->iy_base_vals[ Env_nproc_y(     env ) ];

  Assert( quan->iy_base_vals[ Env_proc_y_this( env )+1 ] -
          quan->iy_base_vals[ Env_proc_y_this( env )   ] == dims.ncell_y );

} /*---Quantities_init_decomp_---*/

/*===========================================================================*/
/*---Pseudo-destructor for Quantities struct---*/

void Quantities_destroy( Quantities* quan )
{
  /*---Deallocate arrays---*/

  Pointer_destroy( & quan->a_from_m );
  Pointer_destroy( & quan->m_from_a );

  free_host_int( quan->ix_base_vals );
  free_host_int( quan->iy_base_vals );

  quan->ix_base_vals = NULL;
  quan->iy_base_vals = NULL;

} /*---Quantities_destroy---*/

/*===========================================================================*/
/*---Flops cost of solve per element---*/

double Quantities_flops_per_solve( const Dimensions dims )
{
  return 3. + 3. * NDIM;
}

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_quantities_testing_c_h_---*/

/*---------------------------------------------------------------------------*/
