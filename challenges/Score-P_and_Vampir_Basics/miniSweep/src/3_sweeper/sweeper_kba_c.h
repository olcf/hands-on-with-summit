/*---------------------------------------------------------------------------*/
/*!
 * \file   sweeper_kba_c.h
 * \author Wayne Joubert
 * \date   Tue Jan 28 16:37:41 EST 2014
 * \brief  Definitions for performing a sweep, kba version.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _sweeper_kba_c_h_
#define _sweeper_kba_c_h_

#include "types.h"
#include "env.h"
#include "pointer.h"
#include "definitions.h"
#include "quantities.h"
#include "array_accessors.h"
#include "array_operations.h"
#include "stepscheduler_kba.h"
#include "sweeper_kba.h"

#include "sweeper_kba_kernels.h"

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
  /*====================*/
  /*---Declarations---*/
  /*====================*/

  Bool_t is_face_comm_async = Arguments_consume_int_or_default( args,
                                           "--is_face_comm_async", Bool_true );

  Insist( dims.ncell_x > 0 ?
                "Currently required that all spatial blocks be nonempty" : 0 );
  Insist( dims.ncell_y > 0 ?
                "Currently required that all spatial blocks be nonempty" : 0 );
  Insist( dims.ncell_z > 0 ?
                "Currently required that all spatial blocks be nonempty" : 0 );

  /*====================*/
  /*---Set up number of kba blocks---*/
  /*====================*/

  sweeper->nblock_z = Arguments_consume_int_or_default( args, "--nblock_z", 1);

  Insist( sweeper->nblock_z > 0 ? "Invalid z blocking factor supplied" : 0 );
  Insist( dims.ncell_z % sweeper->nblock_z == 0
                  ? "Currently require all blocks have same z dimension" : 0 );

  const int dims_b_ncell_z = dims.ncell_z / sweeper->nblock_z;

  /*====================*/
  /*---Set up number of octant threads---*/
  /*====================*/

  sweeper->nthread_octant
              = Arguments_consume_int_or_default( args, "--nthread_octant", 1);

  /*---Require a power of 2 between 1 and 8 inclusive---*/
  Insist( sweeper->nthread_octant>0 && sweeper->nthread_octant<=NOCTANT
          && ((sweeper->nthread_octant&(sweeper->nthread_octant-1))==0)
                                       ? "Invalid thread count supplied" : 0 );
  /*---Don't allow threading in cases where it doesn't make sense---*/
  Insist( sweeper->nthread_octant==1 || IS_USING_OPENMP_THREADS
                                     || IS_USING_OPENMP_TASKS
                                     || Env_cuda_is_using_device( env ) ?
          "Threading not allowed for this case" : 0 );

  sweeper->noctant_per_block = sweeper->nthread_octant;
  sweeper->nblock_octant     = NOCTANT / sweeper->noctant_per_block;

  /*====================*/
  /*---Set up number of semiblock steps---*/
  /*====================*/

  /*---Note special case in which not necessary to semiblock in z---*/

  const int nsemiblock_default = sweeper->nthread_octant==8 &&
                                 sweeper->nblock_z % 2 == 0 ?
                                 4 : sweeper->nthread_octant;

  sweeper->nsemiblock = Arguments_consume_int_or_default(
                                    args, "--nsemiblock", nsemiblock_default );

  Insist( sweeper->nsemiblock>0 && sweeper->nsemiblock<=NOCTANT
          && ((sweeper->nsemiblock&(sweeper->nsemiblock-1))==0)
                                ? "Invalid semiblock count supplied" : 0 );
  Insist( ( sweeper->nsemiblock >= sweeper->nthread_octant ||
            (sweeper->nthread_octant==8 && sweeper->nblock_z % 2 == 0
                                        && sweeper->nsemiblock==4) ||
            IS_USING_OPENMP_VO_ATOMIC )
         ? "Incomplete set of semiblock steps requires atomic vo update" : 0 );

  /*====================*/
  /*---Set up size of subblocks---*/
  /*====================*/

  const int ncell_x_per_subblock_default = sweeper->nsemiblock >= 2 ?
                                           (dims.ncell_x+1) / 2 :
                                            dims.ncell_x;

  const int ncell_y_per_subblock_default = sweeper->nsemiblock >= 4 ?
                                           (dims.ncell_y+1) / 2 :
                                            dims.ncell_y;

  const int ncell_z_per_subblock_default = sweeper->nsemiblock >= 8 ?
                                           (dims_b_ncell_z+1) / 2 :
                                            dims_b_ncell_z;

  sweeper->ncell_x_per_subblock = Arguments_consume_int_or_default(
               args, "--ncell_x_per_subblock", ncell_x_per_subblock_default );
  Insist( sweeper->ncell_x_per_subblock>0 ?
                                        "Invalid subblock size supplied" : 0 );

  sweeper->ncell_y_per_subblock = Arguments_consume_int_or_default(
               args, "--ncell_y_per_subblock", ncell_y_per_subblock_default );
  Insist( sweeper->ncell_y_per_subblock>0 ?
                                        "Invalid subblock size supplied" : 0 );

  sweeper->ncell_z_per_subblock = Arguments_consume_int_or_default(
               args, "--ncell_z_per_subblock", ncell_z_per_subblock_default );
  Insist( sweeper->ncell_z_per_subblock>0 ?
                                        "Invalid subblock size supplied" : 0 );

  /*====================*/
  /*---Set up dims structs---*/
  /*====================*/

  sweeper->dims = dims;

  sweeper->dims_b = sweeper->dims;
  sweeper->dims_b.ncell_z = dims_b_ncell_z;

  sweeper->dims_g = sweeper->dims;
  sweeper->dims_g.ncell_x = quan->ncell_x_g;
  sweeper->dims_g.ncell_y = quan->ncell_y_g;

  /*====================*/
  /*---Set up number of energy threads---*/
  /*====================*/

  sweeper->nthread_e
                   = Arguments_consume_int_or_default( args, "--nthread_e", 1);

  Insist( sweeper->nthread_e > 0 ? "Invalid thread count supplied." : 0 );
  /*---Don't allow threading in cases where it doesn't make sense---*/
  Insist( sweeper->nthread_e==1 || IS_USING_OPENMP_THREADS
                                || IS_USING_OPENMP_TASKS
                                || Env_cuda_is_using_device( env ) ?
          "Threading not allowed for this case" : 0 );

  /*====================*/
  /*---Set up number of spatial threads---*/
  /*====================*/

  if( IS_USING_OPENMP_TASKS )
  {
    /*---TODO: put this logic, repeated in Sweeper_sweep_semiblock etc.,
         in a common place---*/

    const Bool_t is_semiblocked_x = sweeper->nsemiblock > (1<<0);
    const Bool_t is_semiblocked_y = sweeper->nsemiblock > (1<<1);
    const Bool_t is_semiblocked_z = sweeper->nsemiblock > (1<<2);

    const int ncell_x_semiblock_up2 = is_semiblocked_x ?
                                      ( sweeper->dims_b.ncell_x + 1 ) / 2 :
                                        sweeper->dims_b.ncell_x;
    const int ncell_y_semiblock_up2 = is_semiblocked_y ?
                                      ( sweeper->dims_b.ncell_y + 1 ) / 2 :
                                        sweeper->dims_b.ncell_y;
    const int ncell_z_semiblock_up2 = is_semiblocked_z ?
                                      ( sweeper->dims_b.ncell_z + 1 ) / 2 :
                                        sweeper->dims_b.ncell_z;

    sweeper->nthread_x = iceil(ncell_x_semiblock_up2,
                               sweeper->ncell_x_per_subblock);
    sweeper->nthread_y = iceil(ncell_y_semiblock_up2,
                               sweeper->ncell_y_per_subblock);
    sweeper->nthread_z = iceil(ncell_z_semiblock_up2,
                               sweeper->ncell_z_per_subblock);
  }
  else
  {
    sweeper->nthread_x = 1;

    sweeper->nthread_y
                   = Arguments_consume_int_or_default( args, "--nthread_y", 1);

    Insist( sweeper->nthread_y > 0 ? "Invalid thread count supplied." : 0 );
    /*---Don't allow threading in cases where it doesn't make sense---*/
    Insist( sweeper->nthread_y==1 || IS_USING_OPENMP_THREADS
                                  || IS_USING_OPENMP_TASKS
                                  || Env_cuda_is_using_device( env ) ?
            "Threading not allowed for this case" : 0 );
    Insist( sweeper->nthread_y==1 || ! IS_USING_OPENMP_TASKS ?
            "Spatial threading must be defined via subblock sizes." : 0 );

    sweeper->nthread_z
                   = Arguments_consume_int_or_default( args, "--nthread_z", 1);

    Insist( sweeper->nthread_z > 0 ? "Invalid thread count supplied." : 0 );
    /*---Don't allow threading in cases where it doesn't make sense---*/
    Insist( sweeper->nthread_z==1 || IS_USING_OPENMP_THREADS
                                  || IS_USING_OPENMP_TASKS
                                  || Env_cuda_is_using_device( env ) ?
            "Threading not allowed for this case" : 0 );
    Insist( sweeper->nthread_z==1 || ! IS_USING_OPENMP_TASKS ?
            "Spatial threading must be defined via subblock sizes." : 0 );
  }

  /*====================*/
  /*---Set up step scheduler---*/
  /*====================*/

  StepScheduler_create( &(sweeper->stepscheduler),
                              sweeper->nblock_z, sweeper->nblock_octant, env );

  /*====================*/
  /*---Set up amu threads---*/
  /*====================*/

  Insist( NU * 1 > 0 );
  Insist(      Sweeper_nthread_u( sweeper, env ) > 0 );
  Insist( NU % Sweeper_nthread_u( sweeper, env ) == 0 );
  Insist( Sweeper_nthread_a( sweeper, env ) > 0 );
  if( ! IS_USING_MIC )
  {
    Insist( Sweeper_nthread_a( sweeper, env ) %
            Sweeper_nthread_u( sweeper, env ) == 0 );
    Insist( Sweeper_nthread_a( sweeper, env ) ==
            Sweeper_nthread_u( sweeper, env ) *
            Sweeper_nthread_m( sweeper, env ) );
  }
  if( IS_USING_MIC )
  {
    /*---For alignment, make this assumption.  For user case, assume this
         may mean some padding---*/
    Insist( dims.na % VEC_LEN == 0 );
  }

  /*====================*/
  /*---Allocate arrays---*/
  /*====================*/

  sweeper->vilocal_host_ = Env_cuda_is_using_device( env ) ?
                           ( (P*) NULL ) :
                           malloc_host_P( Sweeper_nvilocal_( sweeper, env ) );

  sweeper->vslocal_host_ = Env_cuda_is_using_device( env ) ?
                           ( (P*) NULL ) :
                           malloc_host_P( Sweeper_nvslocal_( sweeper, env ) );

  sweeper->volocal_host_ = Env_cuda_is_using_device( env ) ?
                           ( (P*) NULL ) :
                           malloc_host_P( Sweeper_nvolocal_( sweeper, env ) );

  /*====================*/
  /*---Allocate faces---*/
  /*====================*/

  Faces_create( &(sweeper->faces), sweeper->dims_b,
                sweeper->noctant_per_block, is_face_comm_async, env );
}

/*===========================================================================*/
/*---Pseudo-destructor for Sweeper struct---*/

void Sweeper_destroy( Sweeper* sweeper,
                      Env*     env )
{
  /*====================*/
  /*---Deallocate arrays---*/
  /*====================*/

  if( ! Env_cuda_is_using_device( env ) )
  {
    if( sweeper->vilocal_host_ )
    {
      free_host_P( sweeper->vilocal_host_ );
    }
    if( sweeper->vslocal_host_ )
    {
      free_host_P( sweeper->vslocal_host_ );
    }
    if( sweeper->volocal_host_ )
    {
      free_host_P( sweeper->volocal_host_ );
    }
    sweeper->vilocal_host_ = NULL;
    sweeper->vslocal_host_ = NULL;
    sweeper->volocal_host_ = NULL;
  }

  /*====================*/
  /*---Deallocate faces---*/
  /*====================*/

  Faces_destroy( &(sweeper->faces) );

  /*====================*/
  /*---Terminate scheduler---*/
  /*====================*/

  StepScheduler_destroy( &( sweeper->stepscheduler ) );
}

/*===========================================================================*/
/*---Extract SweeperLite from Sweeper---*/

SweeperLite Sweeper_sweeperlite( Sweeper* sweeper )
{
  SweeperLite sweeperlite;

  sweeperlite.vilocal_host_ = sweeper->vilocal_host_;
  sweeperlite.vslocal_host_ = sweeper->vslocal_host_;
  sweeperlite.volocal_host_ = sweeper->volocal_host_;

  sweeperlite.dims   = sweeper->dims;
  sweeperlite.dims_b = sweeper->dims_b;
  sweeperlite.dims_g = sweeper->dims_g;

  sweeperlite.nthread_e      = sweeper->nthread_e;
  sweeperlite.nthread_octant = sweeper->nthread_octant;
  sweeperlite.nthread_x      = sweeper->nthread_x;
  sweeperlite.nthread_y      = sweeper->nthread_y;
  sweeperlite.nthread_z      = sweeper->nthread_z;

  sweeperlite.nblock_z             = sweeper->nblock_z;
  sweeperlite.nblock_octant        = sweeper->nblock_octant;
  sweeperlite.noctant_per_block    = sweeper->noctant_per_block;
  sweeperlite.nsemiblock           = sweeper->nsemiblock;
  sweeperlite.ncell_x_per_subblock = sweeper->ncell_x_per_subblock;
  sweeperlite.ncell_y_per_subblock = sweeper->ncell_y_per_subblock;
  sweeperlite.ncell_z_per_subblock = sweeper->ncell_z_per_subblock;

#ifdef USE_OPENMP_TASKS
  /*---Mark these as not yet properly initialized---*/
  sweeperlite.thread_e = -1;
  sweeperlite.thread_octant = -1;
  sweeperlite.thread_x = -1;
  sweeperlite.thread_y = -1;
  sweeperlite.thread_z = -1;
  /*---(Arbitrarily) point to the base of the current sweeper struct---*/
  /*---NOTE: will break if sweeperlite is used after sweeper destroyed---*/
  sweeperlite.task_dependency = (char*)sweeper;
#endif

  return sweeperlite;
}

/*===========================================================================*/
/*---Adapter function to launch the sweep block kernel---*/

static void Sweeper_sweep_block_adapter(
  Sweeper*               sweeper,
        P* __restrict__  vo,
  const P* __restrict__  vi,
        P* __restrict__  facexy,
        P* __restrict__  facexz,
        P* __restrict__  faceyz,
  const P* __restrict__  a_from_m,
  const P* __restrict__  m_from_a,
  int                    step,
  const Quantities*      quan,
  Bool_t                 proc_x_min,
  Bool_t                 proc_x_max,
  Bool_t                 proc_y_min,
  Bool_t                 proc_y_max,
  StepInfoAll            stepinfoall,
  unsigned long int      do_block_init,
  Env*                   env )
{
  /*---Create lightweight version of Sweeper class that uses less GPU mem---*/

  SweeperLite sweeperlite = Sweeper_sweeperlite( sweeper );

  /*---Call sweep block implementation function---*/

  if( Env_cuda_is_using_device( env ) )
  {
    Sweeper_sweep_block_impl_global
#ifdef USE_CUDA
                 <<< dim3( Sweeper_nthreadblock( sweeper, 0, env ),
                           Sweeper_nthreadblock( sweeper, 1, env ),
                           Sweeper_nthreadblock( sweeper, 2, env ) ),
                     dim3( Sweeper_nthread_in_threadblock( sweeper, 0, env ),
                           Sweeper_nthread_in_threadblock( sweeper, 1, env ),
                           Sweeper_nthread_in_threadblock( sweeper, 2, env ) ),
                     Sweeper_shared_size_( sweeper, env ),
                     Env_cuda_stream_kernel_faces( env )
                 >>>
#endif
                            ( sweeperlite,
                              vo,
                              vi,
                              facexy,
                              facexz,
                              faceyz,
                              a_from_m,
                              m_from_a,
                              step,
                              *quan,
                              proc_x_min,
                              proc_x_max,
                              proc_y_min,
                              proc_y_max,
                              stepinfoall,
                              do_block_init );
    Assert( Env_cuda_last_call_succeeded() );
  }
  else
  {
#ifdef USE_OPENMP_THREADS
#pragma omp parallel num_threads( sweeper->nthread_e * sweeper->nthread_octant \
                                * sweeper->nthread_y * sweeper->nthread_z )
  {
#endif

    Sweeper_sweep_block_impl( sweeperlite,
                              vo,
                              vi,
                              facexy,
                              facexz,
                              faceyz,
                              a_from_m,
                              m_from_a,
                              step,
                              *quan,
                              proc_x_min,
                              proc_x_max,
                              proc_y_min,
                              proc_y_max,
                              stepinfoall,
                              do_block_init );

#ifdef USE_OPENMP_THREADS
  } /*---OPENMP---*/
#endif
  } /*---if else---*/
}

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
  Env*                   env )
{
  /*---Declarations---*/

  const int proc_x = Env_proc_x_this( env );
  const int proc_y = Env_proc_y_this( env );

  const int noctant_per_block = sweeper->noctant_per_block;

  StepInfoAll stepinfoall;  /*---But only use noctant_per_block values---*/

  int octant_in_block = 0;

  int semiblock_step = 0;

  unsigned long int do_block_init = 0;

  /*---Precalculate stepinfo for required octants---*/

  for( octant_in_block=0; octant_in_block<sweeper->noctant_per_block;
                                                            ++octant_in_block )
  {
    stepinfoall.stepinfo[octant_in_block] = StepScheduler_stepinfo(
      &(sweeper->stepscheduler), step, octant_in_block, proc_x, proc_y );

  }

  /*---Precalculate initialization schedule---*/
  /*---Determine whether this is the first calculation for this sweep step
       and semiblock step - in which case set values rather than add values---*/

  for( semiblock_step=0; semiblock_step<sweeper->nsemiblock; ++semiblock_step )
  {
#pragma novector
    for( octant_in_block=0; octant_in_block<sweeper->noctant_per_block;
                                                            ++octant_in_block )
    {
      const StepInfo stepinfo = stepinfoall.stepinfo[octant_in_block];
      if( stepinfo.is_active )
      {
        const Bool_t is_semiblock_x_lo = is_semiblock_min_when_semiblocked(
            sweeper->nsemiblock, semiblock_step,
            DIM_X, Dir_x( stepinfo.octant ) );
        const Bool_t is_semiblock_min_x = is_semiblock_x_lo ||
                             ! is_axis_semiblocked(sweeper->nsemiblock, DIM_X);


        const Bool_t is_semiblock_y_lo = is_semiblock_min_when_semiblocked(
            sweeper->nsemiblock, semiblock_step,
            DIM_Y, Dir_y( stepinfo.octant ) );
        const Bool_t is_semiblock_min_y = is_semiblock_y_lo ||
                             ! is_axis_semiblocked(sweeper->nsemiblock, DIM_Y);


        const Bool_t is_semiblock_z_lo = is_semiblock_min_when_semiblocked(
            sweeper->nsemiblock, semiblock_step,
            DIM_Z, Dir_z( stepinfo.octant ) );
        const Bool_t is_semiblock_min_z = is_semiblock_z_lo ||
                             ! is_axis_semiblocked(sweeper->nsemiblock, DIM_Z);

        /*---Which semiblock is being processed, according to a uniform
             direction-independent numbering scheme---*/

        const int semiblock_num = ( is_semiblock_min_x ? 0 : 1 ) + 2 * (
                                  ( is_semiblock_min_y ? 0 : 1 ) + 2 * (
                                  ( is_semiblock_min_z ? 0 : 1 ) ));

        /*---Update the running tally of whether this semiblock of this
             block has been initialized yet---*/

        if( ! ( is_block_init[ stepinfo.block_z ] & ( 1 << semiblock_num ) ) )
        {
          do_block_init |= ( ((unsigned long int)1) <<
                             ( octant_in_block + noctant_per_block *
                               semiblock_step ) );
          is_block_init[ stepinfo.block_z ] |= ( 1 << semiblock_num );
        }
      }
    } /*---octant_in_block---*/
  } /*---semiblock---*/

  /*---Call kernel adapter---*/

  Sweeper_sweep_block_adapter( sweeper,
                               Pointer_active( vo ),
                               Pointer_active( vi ),
                               Pointer_active( facexy ),
                               Pointer_active( facexz ),
                               Pointer_active( faceyz ),
                               Pointer_const_active( a_from_m ),
                               Pointer_const_active( m_from_a ),
                               step,
                               quan,
                               proc_x==0,
                               proc_x==Env_nproc_x( env )-1,
                               proc_y==0,
                               proc_y==Env_nproc_y( env )-1,
                               stepinfoall,
                               do_block_init,
                               env);
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

  const int nblock_z = sweeper->nblock_z;

  const int nstep = StepScheduler_nstep( &(sweeper->stepscheduler) );
  int step = -1;

  const size_t size_state_block = Dimensions_size_state( sweeper->dims, NU )
                                                                   / nblock_z;

  Bool_t* is_block_init = (Bool_t*) malloc( nblock_z * sizeof( Bool_t ) );

  int i = 0;

  for( i=0; i<nblock_z; ++i )
  {
    is_block_init[i] = 0;
  }

  /*---Initialize result array to zero if needed---*/

#ifdef USE_OPENMP_VO_ATOMIC
  initialize_state_zero( Pointer_h( vo ), sweeper->dims, NU );
  Pointer_update_d_stream( vo, Env_cuda_stream_kernel_faces( env ) );
#endif

  /*--------------------*/
  /*---Loop over kba parallel steps---*/
  /*--------------------*/

  /*---Extra step at begin/end to fill/drain async pipeline---*/

  for( step=0-1; step<nstep+1; ++step )
  {
    const Bool_t is_sweep_step = step>=0 && step<nstep;

    /*---Pointers to single active block of state vector---*/

    Pointer vi_b = Pointer_null();
    Pointer vo_b = Pointer_null();

    int i = 0;

    /*---Pick up needed face pointers---*/

    /*=========================================================================
    =    Order is important here.
    =    The _r face for a step must match the _c face for the next step.
    =    The _s face for a step must match the _c face for the prev step.
    =========================================================================*/

    Pointer* facexy = Faces_facexy_step( &(sweeper->faces), step );
    Pointer* facexz = Faces_facexz_step( &(sweeper->faces), step );
    Pointer* faceyz = Faces_faceyz_step( &(sweeper->faces), step );

    /*=========================================================================
    =    Faces are triple buffered via a circular buffer of face arrays.
    =    The following shows the pattern of face usage over a step:
    =
    =                         step:     ...    i    i+1   i+2   i+3   ...
    =    ------------------------------------------------------------------
    =    Recv face for this step wait   ...  face0 face1 face2 face0  ...
    =    Recv face for next step start  ...  face1 face2 face0 face1  ...
    =    Compute this step using face   ...  face0 face1 face2 face0  ...
    =    Send face from last step wait  ...  face2 face0 face1 face2  ...
    =    Send face from this step start ...  face0 face1 face2 face0  ...
    =========================================================================*/

    /*====================*/
    /*---Recv face via MPI WAIT (i)---*/
    /*====================*/

    if( is_sweep_step &&  Faces_is_face_comm_async( &(sweeper->faces)) )
    {
      Faces_recv_faces_end( &(sweeper->faces), &(sweeper->stepscheduler),
                            sweeper->dims_b, step-1, env );
    }

    /*====================*/
    /*---Send face to device START (i)---*/
    /*---Send face to device WAIT (i)---*/
    /*====================*/

    if( is_sweep_step )
    {
      if( step == 0 )
      {
        Pointer_update_d_stream( facexy, Env_cuda_stream_kernel_faces( env ) );
      }
      Pointer_update_d_stream(   facexz, Env_cuda_stream_kernel_faces( env ) );
      Pointer_update_d_stream(   faceyz, Env_cuda_stream_kernel_faces( env ) );
    }
    Env_cuda_stream_wait( env, Env_cuda_stream_kernel_faces( env ) );

    /*====================*/
    /*---Recv face via MPI START (i+1)---*/
    /*====================*/

    if( is_sweep_step &&  Faces_is_face_comm_async( &(sweeper->faces)) )
    {
      Faces_recv_faces_start( &(sweeper->faces), &(sweeper->stepscheduler),
                            sweeper->dims_b, step, env );
    }

    /*====================*/
    /*---Perform the sweep on the block START (i)---*/
    /*====================*/

    if( is_sweep_step )
    {
      Sweeper_sweep_block( sweeper, vo, vi, is_block_init,
                           facexy, facexz, faceyz,
                           & quan->a_from_m, & quan->m_from_a,
                           step, quan, env );
    }

    /*====================*/
    /*---Send block to device START (i+1)---*/
    /*====================*/

    for( i=0; i<2; ++i )
    {
      /*---Determine blocks needing transfer, counting from top/bottom z---*/
      /*---NOTE: for case of one octant thread, can speed this up by only
           send/recv of one block per step, not two---*/

      const int stept = step + 1;
      const int    block_to_send[2] = {                                stept,
                                        ( nblock_z-1 ) -               stept };
      const Bool_t do_block_send[2] = { block_to_send[0] <  nblock_z/2,
                                        block_to_send[1] >= nblock_z/2 };
      Assert( nstep >= nblock_z );  /*---Sanity check---*/
      if( do_block_send[i] )
      {
        Pointer_create_alias(    &vi_b, vi, size_state_block * block_to_send[i],
                                            size_state_block );
        Pointer_update_d_stream( &vi_b, Env_cuda_stream_send_block( env ) );
        Pointer_destroy(         &vi_b );

        /*---Initialize result array to zero if needed---*/
        /*---NOTE: this is not performance-optimal---*/
#ifdef USE_OPENMP_VO_ATOMIC
        Pointer_create_alias(    &vo_b, vi, size_state_block * block_to_send[i],
                                          size_state_block );
        initialize_state_zero( Pointer_h( &vo_b ), sweeper->dims, NU );
        Pointer_update_d_stream( &vo_b, Env_cuda_stream_send_block( env ) );
        Pointer_destroy(         &vo_b );
#endif
      }
    }

    /*====================*/
    /*---Recv block from device START (i-1)---*/
    /*====================*/

    for( i=0; i<2; ++i )
    {
      /*---Determine blocks needing transfer, counting from top/bottom z---*/
      /*---NOTE: for case of one octant thread, can speed this up by only
           send/recv of one block per step, not two---*/

      const int stept = step - 1;
      const int    block_to_recv[2] = { ( nblock_z-1 ) - ( nstep-1 - stept ),
                                                         ( nstep-1 - stept ) };
      const Bool_t do_block_recv[2] = { block_to_recv[0] >= nblock_z/2,
                                        block_to_recv[1] <  nblock_z/2 };
      Assert( nstep >= nblock_z );  /*---Sanity check---*/
      if( do_block_recv[i] )
      {
        Pointer_create_alias(    &vo_b, vo, size_state_block * block_to_recv[i],
                                            size_state_block );
        Pointer_update_h_stream( &vo_b, Env_cuda_stream_recv_block( env ) );
        Pointer_destroy(         &vo_b );
      }
    }

    /*====================*/
    /*---Send block to device WAIT (i+1)---*/
    /*---Recv block from device WAIT (i-1)---*/
    /*====================*/

    Env_cuda_stream_wait( env, Env_cuda_stream_send_block( env ) );
    Env_cuda_stream_wait( env, Env_cuda_stream_recv_block( env ) );

    /*====================*/
    /*---Send face via MPI WAIT (i-1)---*/
    /*====================*/

    if( is_sweep_step && Faces_is_face_comm_async( &(sweeper->faces)) )
    {
      Faces_send_faces_end( &(sweeper->faces), &(sweeper->stepscheduler),
                            sweeper->dims_b, step-1, env );
    }

    /*====================*/
    /*---Perform the sweep on the block WAIT (i)---*/
    /*====================*/

    Env_cuda_stream_wait( env, Env_cuda_stream_kernel_faces( env ) );

    /*====================*/
    /*---Recv face from device START (i)---*/
    /*---Recv face from device WAIT (i)---*/
    /*====================*/

    if( is_sweep_step )
    {
      if( step == nstep-1 )
      {
        Pointer_update_h_stream( facexy, Env_cuda_stream_kernel_faces( env ) );
      }
      Pointer_update_h_stream(   facexz, Env_cuda_stream_kernel_faces( env ) );
      Pointer_update_h_stream(   faceyz, Env_cuda_stream_kernel_faces( env ) );
    }
    Env_cuda_stream_wait( env, Env_cuda_stream_kernel_faces( env ) );

    /*====================*/
    /*---Send face via MPI START (i)---*/
    /*====================*/

    if( is_sweep_step && Faces_is_face_comm_async( &(sweeper->faces)) )
    {
      Faces_send_faces_start( &(sweeper->faces), &(sweeper->stepscheduler),
                            sweeper->dims_b, step, env );
    }

    /*====================*/
    /*---Communicate faces (synchronous)---*/
    /*====================*/

    if( is_sweep_step && ! Faces_is_face_comm_async( &(sweeper->faces)) )
    {
      Faces_communicate_faces( &(sweeper->faces), &(sweeper->stepscheduler),
                            sweeper->dims_b, step, env );
    }

  } /*---step---*/

  /*---Increment message tag---*/

  Env_increment_tag( env, sweeper->noctant_per_block );

  /*---Finish---*/

  free( (void*) is_block_init );

} /*---sweep---*/

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_sweeper_kba_c_h_---*/

/*---------------------------------------------------------------------------*/
