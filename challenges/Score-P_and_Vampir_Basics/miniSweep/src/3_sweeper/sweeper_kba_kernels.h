/*---------------------------------------------------------------------------*/
/*!
 * \file   sweeper_kba_kernels.h
 * \author Wayne Joubert
 * \date   Tue Jan 28 16:37:41 EST 2014
 * \brief  sweeper_kba, code for comp. kernel.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _sweeper_kba_kernels_h_
#define _sweeper_kba_kernels_h_

#include "types_kernels.h"
#include "env_kernels.h"
#include "definitions_kernels.h"
#include "dimensions_kernels.h"
#include "pointer_kernels.h"
#include "quantities_kernels.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Set up enums---*/

#ifdef USE_OPENMP_THREADS
  enum{ IS_USING_OPENMP_THREADS = 1 };
#else
  enum{ IS_USING_OPENMP_THREADS = 0 };
#endif

#ifdef USE_OPENMP_VO_ATOMIC
  enum{ IS_USING_OPENMP_VO_ATOMIC = 1 };
#else
  enum{ IS_USING_OPENMP_VO_ATOMIC = 0 };
#endif

#ifdef USE_OPENMP_TASKS
  enum{ IS_USING_OPENMP_TASKS = 1};
#else
  enum{ IS_USING_OPENMP_TASKS = 0};
#endif

/*---NOTE: these should NOT be accessed outside of the Sweeper pseudo-class---*/

enum{ NTHREAD_DEVICE_U = VEC_LEN <= NU                  ? VEC_LEN :
                         ( NU * NM * 1 <= VEC_LEN * 1 ) ? NU :
                           NM - 0 == 16 && NU - 0 == 4  ?  2 :
                                                          NU };
enum{ NTHREAD_DEVICE_M = VEC_LEN / NTHREAD_DEVICE_U };
enum{ NTHREAD_DEVICE_A = NTHREAD_DEVICE_U * NTHREAD_DEVICE_M };

#ifdef __CUDA_ARCH__
  enum{ NTHREAD_A = NTHREAD_DEVICE_A };
  enum{ NTHREAD_M = NTHREAD_DEVICE_M };
  enum{ NTHREAD_U = NTHREAD_DEVICE_U };
#else
#ifdef __MIC__
  enum{ NTHREAD_A = VEC_LEN * 4 }; /*---tuning parameter---*/
  enum{ NTHREAD_M = NM };
  enum{ NTHREAD_U = NU };
#else
  enum{ NTHREAD_A = NTHREAD_DEVICE_A };
  enum{ NTHREAD_M = NTHREAD_DEVICE_M };
  enum{ NTHREAD_U = NTHREAD_DEVICE_U };
#endif
#endif

/*===========================================================================*/
/*---Lightweight version of Sweeper class for sending to device---*/

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
#ifdef USE_OPENMP_TASKS
  int              thread_e;
  int              thread_octant;
  int              thread_x;
  int              thread_y;
  int              thread_z;
  char*            task_dependency;
#endif
} SweeperLite;

/*===========================================================================*/
/*---Thread indexers---*/

TARGET_HD static inline int Sweeper_thread_e( const SweeperLite* sweeper )
{
#ifdef USE_OPENMP_TASKS
  Assert(sweeper->thread_e >= 0);
  return sweeper->thread_e;
#else
#ifdef __CUDA_ARCH__
  return Env_cuda_threadblock( 0 );
#else
  Assert( sweeper->nthread_e *
          sweeper->nthread_octant *
          sweeper->nthread_y *
          sweeper->nthread_z == 1 || Env_omp_in_parallel() );
  return Env_omp_thread() % sweeper->nthread_e;
#endif
#endif
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline int Sweeper_thread_octant( const SweeperLite* sweeper )
{
#ifdef USE_OPENMP_TASKS
  Assert(sweeper->thread_octant >= 0);
  return sweeper->thread_octant;
#else
#ifdef __CUDA_ARCH__
  return Env_cuda_thread_in_threadblock( 1 );
#else
  Assert( sweeper->nthread_e *
          sweeper->nthread_octant *
          sweeper->nthread_y *
          sweeper->nthread_z == 1 || Env_omp_in_parallel() );
  return ( Env_omp_thread() / sweeper->nthread_e )
                            % sweeper->nthread_octant;
#endif
#endif
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline int Sweeper_thread_x( const SweeperLite* sweeper )
{
#ifdef USE_OPENMP_TASKS
  Assert(sweeper->thread_x >= 0);
  return sweeper->thread_x;
#else
  return 0;
#endif
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline int Sweeper_thread_y( const SweeperLite* sweeper )
{
#ifdef USE_OPENMP_TASKS
  Assert(sweeper->thread_y >= 0);
  return sweeper->thread_y;
#else
#ifdef __CUDA_ARCH__
  return Env_cuda_thread_in_threadblock( 2 ) % sweeper->nthread_y ;
#else
  Assert( sweeper->nthread_e *
          sweeper->nthread_octant *
          sweeper->nthread_y *
          sweeper->nthread_z == 1 || Env_omp_in_parallel() );
  return ( Env_omp_thread() / ( sweeper->nthread_e *
                                sweeper->nthread_octant )
                            %   sweeper->nthread_y );
#endif
#endif
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline int Sweeper_thread_z( const SweeperLite* sweeper )
{
#ifdef USE_OPENMP_TASKS
  Assert(sweeper->thread_z >= 0);
  return sweeper->thread_z;
#else
#ifdef __CUDA_ARCH__
  return Env_cuda_thread_in_threadblock( 2 ) / sweeper->nthread_y;
#else
  Assert( sweeper->nthread_e *
          sweeper->nthread_octant *
          sweeper->nthread_y *
          sweeper->nthread_z == 1 || Env_omp_in_parallel() );
  return ( Env_omp_thread() / ( sweeper->nthread_e *
                                sweeper->nthread_octant *
                                sweeper->nthread_y )
                            %   sweeper->nthread_z );
#endif
#endif
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline int Sweeper_thread_a( const SweeperLite* sweeper )
{
#ifdef __CUDA_ARCH__
  return Env_cuda_thread_in_threadblock( 0 );
#else
  return 0;
#endif
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline int Sweeper_thread_m( const SweeperLite* sweeper )
{
#ifdef __CUDA_ARCH__
  return Env_cuda_thread_in_threadblock( 0 ) / NTHREAD_U;
#else
  return 0;
#endif
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline int Sweeper_thread_u( const SweeperLite* sweeper )
{
#ifdef __CUDA_ARCH__
  return Env_cuda_thread_in_threadblock( 0 ) % NTHREAD_U;
#else
  return 0;
#endif
}

/*===========================================================================*/
/*---Thread synchronization---*/

TARGET_HD static inline void Sweeper_sync_octant_threads( SweeperLite* sweeper )
{
#ifdef __CUDA_ARCH__
  /*---NOTE: this may not be needed if these threads are mapped in-warp---*/
  Env_cuda_sync_threadblock();
#else
#ifdef USE_OPENMP_THREADS
if( sweeper->nthread_octant != 1 )
{
#pragma omp barrier
}
#endif
#endif
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline void Sweeper_sync_yz_threads( SweeperLite* sweeper )
{
#ifdef __CUDA_ARCH__
  /*---NOTE: this may not be needed if these threads are mapped in-warp---*/
  Env_cuda_sync_threadblock();
#else
#ifdef USE_OPENMP_THREADS
if( sweeper->nthread_y != 1 || sweeper->nthread_z != 1 )
{
#pragma omp barrier
}
#endif
#endif
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline void Sweeper_sync_amu_threads( SweeperLite* sweeper )
{
#ifdef __CUDA_ARCH__
  /*---NOTE: this may not be needed if these threads are mapped in-warp---*/
  Env_cuda_sync_threadblock();
#else
#ifdef USE_OPENMP_THREADS
/*---amu axes not threaded for openmp case---*/
#endif
#endif
}

/*===========================================================================*/
/*---Select which part of v*local to use for current thread/block---*/

TARGET_HD static inline P* __restrict__ Sweeper_vilocal_this_(
                                                         SweeperLite* sweeper )
{
#ifdef __CUDA_ARCH__
  return ( (P*) Env_cuda_shared_memory() )
    + ( NTHREAD_M *
        NU *
        sweeper->nthread_octant *
     /* sweeper->nthread_x *    (guaranteed is equal to 1) */
        sweeper->nthread_y *
        sweeper->nthread_z ) * 0
    + NTHREAD_M *
      NU *
      ( Sweeper_thread_octant( sweeper ) + sweeper->nthread_octant * (
     /* Sweeper_thread_x(      sweeper ) + sweeper->nthread_x      * ( */
        Sweeper_thread_y(      sweeper ) + sweeper->nthread_y      * (
        Sweeper_thread_z(      sweeper ) + sweeper->nthread_z      * (
        0 ) ) ) )
  ;
#else
  return sweeper->vilocal_host_
    + NTHREAD_M *
      NU *
      ( Sweeper_thread_octant( sweeper ) + sweeper->nthread_octant * (
        Sweeper_thread_x(      sweeper ) + sweeper->nthread_x      * (
        Sweeper_thread_y(      sweeper ) + sweeper->nthread_y      * (
        Sweeper_thread_z(      sweeper ) + sweeper->nthread_z      * (
        Sweeper_thread_e(      sweeper ) + sweeper->nthread_e      * (
        0 ) ) ) ) ) )
  ;
#endif
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline P* __restrict__ Sweeper_vslocal_this_(
                                                         SweeperLite* sweeper )
{
#ifdef __CUDA_ARCH__
  return ( (P*) Env_cuda_shared_memory() )
    + ( NTHREAD_M *
        NU *
        sweeper->nthread_octant *
     /* sweeper->nthread_x *    (guaranteed is equal to 1) */
        sweeper->nthread_y *
        sweeper->nthread_z ) * 2
    + NTHREAD_A *
      NU *
      ( Sweeper_thread_octant( sweeper ) + sweeper->nthread_octant * (
     /* Sweeper_thread_x(      sweeper ) + sweeper->nthread_x      * ( */
        Sweeper_thread_y(      sweeper ) + sweeper->nthread_y      * (
        Sweeper_thread_z(      sweeper ) + sweeper->nthread_z      * (
        0 ) ) ) )
  ;
#else
  return sweeper->vslocal_host_
    + NTHREAD_A *
      NU *
      ( Sweeper_thread_octant( sweeper ) + sweeper->nthread_octant * (
        Sweeper_thread_x(      sweeper ) + sweeper->nthread_x      * (
        Sweeper_thread_y(      sweeper ) + sweeper->nthread_y      * (
        Sweeper_thread_z(      sweeper ) + sweeper->nthread_z      * (
        Sweeper_thread_e(      sweeper ) + sweeper->nthread_e      * (
        0 ) ) ) ) ) )
  ;
#endif
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline P* __restrict__ Sweeper_volocal_this_(
                                                         SweeperLite* sweeper )
{
#ifdef __CUDA_ARCH__
  return ( (P*) Env_cuda_shared_memory() )
    + ( NTHREAD_M *
        NU *
        sweeper->nthread_octant *
     /* sweeper->nthread_x *    (guaranteed is equal to 1) */
        sweeper->nthread_y *
        sweeper->nthread_z ) * 1
    + NTHREAD_M *
      NU *
      ( Sweeper_thread_octant( sweeper ) + sweeper->nthread_octant * (
     /* Sweeper_thread_x(      sweeper ) + sweeper->nthread_x      * ( */
        Sweeper_thread_y(      sweeper ) + sweeper->nthread_y      * (
        Sweeper_thread_z(      sweeper ) + sweeper->nthread_z      * (
        0 ) ) ) )
  ;
#else
  return sweeper->volocal_host_
    + NTHREAD_M *
      NU *
      ( Sweeper_thread_octant( sweeper ) + sweeper->nthread_octant * (
        Sweeper_thread_x(      sweeper ) + sweeper->nthread_x      * (
        Sweeper_thread_y(      sweeper ) + sweeper->nthread_y      * (
        Sweeper_thread_z(      sweeper ) + sweeper->nthread_z      * (
        Sweeper_thread_e(      sweeper ) + sweeper->nthread_e      * (
        0 ) ) ) ) ) )
  ;
#endif
}

/*===========================================================================*/
/*---Helper functions---*/

TARGET_HD static inline Bool_t is_axis_semiblocked(int nsemiblock, int dim)
{
  /* Indicate whether the block is broken into semiblocks along the axis */
  /* Note as we increase nsemiblock, we semiblock in x, then add y, then z */

  Assert( nsemiblock >= 0 && nsemiblock <= NOCTANT );
  Assert( dim >= 0 && dim < NDIM );

  return nsemiblock > (1<<dim);
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline Bool_t is_semiblock_min_when_semiblocked(
                          int nsemiblock, int semiblock_step, int dim, int dir)
{
  /* On this semiblock step for this thread, do we process the lower
     semiblock along the relevant axis.  Only meaningful if is_semiblocked. */

  Assert( nsemiblock >= 0 && nsemiblock <= NOCTANT );
  Assert( semiblock_step >= 0 && semiblock_step < nsemiblock );
  Assert( dim >= 0 && dim < NDIM );
  Assert( dir == DIR_UP || dir == DIR_DN );

  return ( ( semiblock_step & (1<<dim) ) == 0 )  ==  ( dir == DIR_UP );
}

/*---------------------------------------------------------------------------*/

#ifdef USE_OPENMP_TASKS
static inline char* Sweeper_task_dependency( SweeperLite* sweeperlite,
  int thread_x, int thread_y, int thread_z, int thread_e, int thread_octant )
{
  return &( sweeperlite->task_dependency[
    1 + thread_x      + (1+sweeperlite->nthread_x)      * (
    1 + thread_y      + (1+sweeperlite->nthread_y)      * (
    1 + thread_z      + (1+sweeperlite->nthread_z)      * (
    1 + thread_e      + (1+sweeperlite->nthread_e)      * (
    1 + thread_octant + (1+sweeperlite->nthread_octant) * ( 0 ))))) ] );
}
#endif

/*===========================================================================*/
/*---Perform a sweep for a block, implementation---*/

TARGET_HD void Sweeper_sweep_block_impl(
  SweeperLite            sweeper,
  P* __restrict__        vo,
  const P* __restrict__  vi,
  P* __restrict__        facexy,
  P* __restrict__        facexz,
  P* __restrict__        faceyz,
  const P* __restrict__  a_from_m,
  const P* __restrict__  m_from_a,
  int                    step,
  const Quantities       quan,
  Bool_t                 proc_x_min,
  Bool_t                 proc_x_max,
  Bool_t                 proc_y_min,
  Bool_t                 proc_y_max,
  StepInfoAll            stepinfoall,
  unsigned long int      do_block_init );

/*===========================================================================*/
/*---Perform a sweep for a block, implementation, global---*/

TARGET_G void Sweeper_sweep_block_impl_global(
  SweeperLite            sweeper,
        P* __restrict__  vo,
  const P* __restrict__  vi,
        P* __restrict__  facexy,
        P* __restrict__  facexz,
        P* __restrict__  faceyz,
  const P* __restrict__  a_from_m,
  const P* __restrict__  m_from_a,
  int                    step,
  const Quantities       quan,
  Bool_t                 proc_x_min,
  Bool_t                 proc_x_max,
  Bool_t                 proc_y_min,
  Bool_t                 proc_y_max,
  StepInfoAll            stepinfoall,
  unsigned long int      do_block_init );

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_sweeper_kba_kernels_h_---*/

/*---------------------------------------------------------------------------*/
