/*---------------------------------------------------------------------------*/
/*!
 * \file   runner.c
 * \author Wayne Joubert
 * \date   Wed Jan 28 10:11:10 EST 2015
 * \brief  Definitions for tools to perform runs of sweeper.
 * \note   Copyright (C) 2013 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#include <stdio.h>

#include "arguments.h"
#include "env.h"
#include "definitions.h"
#include "dimensions.h"
#include "pointer.h"
#include "quantities.h"
#include "array_operations.h"
#include "sweeper.h"

#include "runner.h"

/*===========================================================================*/
/*---Null object---*/

Runner Runner_null()
{
  Runner result;
  memset( (void*)&result, 0, sizeof(Runner) );
  return result;
}

/*===========================================================================*/
/*---Pseudo-constructor---*/

void Runner_create( Runner* runner )
{
}

/*===========================================================================*/
/*---Pseudo-destructor---*/

void Runner_destroy( Runner* runner )
{
}

/*===========================================================================*/
/*---Perform run---*/

void Runner_run_case( Runner* runner, Arguments* args, Env* env )
{
  /*---Declarations---*/

  Dimensions  dims_g;       /*---dims for entire problem---*/
  Dimensions  dims;         /*---dims for the part on this MPI proc---*/
  Quantities  quan;
  Sweeper     sweeper = Sweeper_null();

  Pointer vi = Pointer_null();
  Pointer vo = Pointer_null();

  runner->normsq     = P_zero();
  runner->normsqdiff = P_zero();

  int iteration   = 0;
  int niterations = 0;

  Timer t1             = 0;
  Timer t2             = 0;

  runner->time       = 0;
  runner->flops      = 0;
  runner->floprate   = 0;
  runner->normsq     = 0;
  runner->normsqdiff = 0;

  /*---Define problem specs---*/

  dims_g.ncell_x = Arguments_consume_int_or_default( args, "--ncell_x",  5 );
  dims_g.ncell_y = Arguments_consume_int_or_default( args, "--ncell_y",  5 );
  dims_g.ncell_z = Arguments_consume_int_or_default( args, "--ncell_z",  5 );
  dims_g.ne   = Arguments_consume_int_or_default( args, "--ne", 30 );
  dims_g.na   = Arguments_consume_int_or_default( args, "--na", 33 );
  niterations = Arguments_consume_int_or_default( args, "--niterations", 1 );
  dims_g.nm   = NM;

  Insist( dims_g.ncell_x > 0 ? "Invalid ncell_x supplied." : 0 );
  Insist( dims_g.ncell_y > 0 ? "Invalid ncell_y supplied." : 0 );
  Insist( dims_g.ncell_z > 0 ? "Invalid ncell_z supplied." : 0 );
  Insist( dims_g.ne > 0      ? "Invalid ne supplied." : 0 );
  Insist( dims_g.nm > 0      ? "Invalid nm supplied." : 0 );
  Insist( dims_g.na > 0      ? "Invalid na supplied." : 0 );
  Insist( niterations >= 0   ? "Invalid iteration count supplied." : 0 );

  /*---Initialize (local) dimensions - domain decomposition---*/

  dims = dims_g;

  dims.ncell_x =
      ( ( Env_proc_x_this( env ) + 1 ) * dims_g.ncell_x ) / Env_nproc_x( env )
    - ( ( Env_proc_x_this( env )     ) * dims_g.ncell_x ) / Env_nproc_x( env );

  dims.ncell_y =
      ( ( Env_proc_y_this( env ) + 1 ) * dims_g.ncell_y ) / Env_nproc_y( env )
    - ( ( Env_proc_y_this( env )     ) * dims_g.ncell_y ) / Env_nproc_y( env );

  /*---Initialize quantities---*/

  Quantities_create( &quan, dims, env );

  /*---Allocate arrays---*/

  Pointer_create( &vi, Dimensions_size_state( dims, NU ),
                                            Env_cuda_is_using_device( env ) );
  Pointer_set_pinned( &vi, Bool_true );
  Pointer_allocate( &vi );

  Pointer_create( &vo, Dimensions_size_state( dims, NU ),
                                            Env_cuda_is_using_device( env ) );
  Pointer_set_pinned( &vo, Bool_true );
  Pointer_allocate( &vo );

  /*---Initialize input state array---*/

  initialize_state( Pointer_h( &vi ), dims, NU, &quan );

  /*---Initialize output state array---*/
  /*---This is not strictly required for the output vector but might
       have a performance effect from pre-touching pages.
  ---*/

  initialize_state_zero( Pointer_h( &vo ), dims, NU );

  /*---Initialize sweeper---*/

  Sweeper_create( &sweeper, dims, &quan, env, args );

  /*---Check that all command line args used---*/

  Insist( Arguments_are_all_consumed( args )
                                          ? "Invalid argument detected." : 0 );

  /*---Call sweeper---*/

  t1 = Env_get_synced_time( env );

  for( iteration=0; iteration<niterations; ++iteration )
  {
    Sweeper_sweep( &sweeper,
                   iteration%2==0 ? &vo : &vi,
                   iteration%2==0 ? &vi : &vo,
                   &quan,
                   env );
  }

  t2 = Env_get_synced_time( env );
  runner->time = t2 - t1;

  /*---Compute flops used---*/

  runner->flops = Env_sum_d( env, niterations *
         ( Dimensions_size_state( dims, NU ) * NOCTANT * 2. * dims.na
         + Dimensions_size_state_angles( dims, NU )
                                        * Quantities_flops_per_solve( dims )
         + Dimensions_size_state( dims, NU ) * NOCTANT * 2. * dims.na ) );

  runner->floprate = runner->time <= (Timer)0 ?
                                   0 : runner->flops / runner->time / 1e9;

  /*---Compute, print norm squared of result---*/

  get_state_norms( Pointer_h( &vi ), Pointer_h( &vo ),
                     dims, NU, &runner->normsq, &runner->normsqdiff, env );

  /*---Deallocations---*/
  Pointer_destroy( &vi );
  Pointer_destroy( &vo );

  Sweeper_destroy( &sweeper, env );
  Quantities_destroy( &quan );
}

/*===========================================================================*/
/*---Utility function: perform two runs, compare results---*/

Bool_t compare_runs( const char* argstring1, const char* argstring2, Env* env )
{
  Arguments args1 = Arguments_null();
  Arguments args2 = Arguments_null();
  Runner  runner1 = Runner_null();
  Runner  runner2 = Runner_null();

  Runner_create( &runner1 );
  Runner_create( &runner2 );

  Arguments_create_from_string( &args1, argstring1 );
  Env_set_values( env, &args1 );

  if( Env_is_proc_master( env ) )
  {
    printf("%s // ", argstring1);
  }
  if( Env_is_proc_active( env ) )
  {
    Runner_run_case( &runner1, &args1, env );
  }

  Arguments_create_from_string( &args2, argstring2 );
  Env_set_values( env, &args2 );

  if( Env_is_proc_master( env ) )
  {
    printf("%s // ", argstring2);
  }
  if( Env_is_proc_active( env ) )
  {
    Runner_run_case( &runner2, &args2, env );
  }

  Bool_t pass = Env_is_proc_master( env ) ?
                runner1.normsqdiff == P_zero() &&
                runner2.normsqdiff == P_zero() &&
                runner1.normsq == runner2.normsq : Bool_false;

  if( Env_is_proc_master( env ) )
  {
    printf("%e %e %e %e // %i %i %i // %s\n",
      runner1.normsqdiff, runner2.normsqdiff,
      runner1.normsq, runner2.normsq,
      runner1.normsq == runner2.normsq,
      runner1.normsqdiff == P_zero(),
      runner2.normsqdiff == P_zero(),
      pass ? "PASS" : "FAIL" );
  }

  Runner_destroy( &runner1 );
  Runner_destroy( &runner2 );

  return pass;
}

/*---------------------------------------------------------------------------*/
