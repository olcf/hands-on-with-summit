/*---------------------------------------------------------------------------*/
/*!
 * \file   sweeper_kba_c_kernels.h
 * \author Wayne Joubert
 * \date   Tue Jan 28 16:37:41 EST 2014
 * \brief  sweeper_kba_c, code for comp. kernel.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _sweeper_kba_c_kernels_h_
#define _sweeper_kba_c_kernels_h_

#include "types_kernels.h"
#include "env_kernels.h"
#include "definitions_kernels.h"
#include "quantities_kernels.h"
#include "array_accessors_kernels.h"
#include "stepscheduler_kba_kernels.h"
#include "sweeper_kba_kernels.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Perform a sweep for a cell---*/

TARGET_HD static inline void Sweeper_sweep_cell(
  SweeperLite* __restrict__      sweeper,
  P* const __restrict__          vo_this,
  const P* const __restrict__    vi_this,
  P* const __restrict__          vilocal,
  P* const __restrict__          vslocal,
  P* const __restrict__          volocal,
  P* const __restrict__          facexy,
  P* const __restrict__          facexz,
  P* const __restrict__          faceyz,
  const P* const __restrict__    a_from_m,
  const P* const __restrict__    m_from_a,
  const Quantities* __restrict__ quan,
  const int                      octant,
  const int                      iz_base,
  const int                      octant_in_block,
  const int                      ie,
  const int                      ix,
  const int                      iy,
  const int                      iz,
  const Bool_t                   do_block_init_this,
  const Bool_t                   is_elt_active )
{
  enum{ NU_PER_THREAD = NU / NTHREAD_U };

  int ia_base = 0;

  const int sweeper_thread_a = Sweeper_thread_a( sweeper );
  const int sweeper_thread_m = Sweeper_thread_m( sweeper );
  const int sweeper_thread_u = Sweeper_thread_u( sweeper );

  /*---For cases when the number of angles or the number of moments is
       too large for the warp size, a 2-D blocking strategy is used
       to keep the granularity of the work to be within warp---*/

  /*---NOTE for computations threaded in angle, the unknowns axis U is
       serial (looped), but for operations threaded in moment,
       some threadiung in U is allowed---*/

  /*====================*/
  /*---Master loop over angle blocks---*/
  /*====================*/

  for( ia_base=0; ia_base<sweeper->dims_b.na; ia_base += NTHREAD_A )
  {
    int im_base = 0;

    for( im_base=0; im_base<NM; im_base += NTHREAD_M )
    {

      /*====================*/
      /*---If needed: reassign threads from angles to moments---*/
      /*====================*/

      if( im_base != 0 )
      {
        Sweeper_sync_amu_threads( sweeper );
      }

      /*====================*/
      /*---Load portion of vi---*/
      /*====================*/

      {
        __assume_aligned( vi_this, ( VEC_LEN < NTHREAD_M*NTHREAD_U ?
                                     VEC_LEN : NTHREAD_M*NTHREAD_U )
                                                                 * sizeof(P) );
        __assume_aligned( vilocal, ( VEC_LEN < NTHREAD_M*NTHREAD_U ?
                                     VEC_LEN : NTHREAD_M*NTHREAD_U )
                                                                 * sizeof(P) );
#ifndef __CUDA_ARCH__
        int sweeper_thread_m = 0;
        int sweeper_thread_u = 0;
        for( sweeper_thread_u=0; sweeper_thread_u<NTHREAD_U;
                                                          ++sweeper_thread_u )
#pragma ivdep
/*
#pragma simd assert, vectorlengthfor( P )
TODO: fix vectorization for this loop.
*/
#pragma simd vectorlengthfor( P )
        for( sweeper_thread_m=0; sweeper_thread_m<NTHREAD_M;
                                                          ++sweeper_thread_m )
#endif
        {
          const int im = im_base + sweeper_thread_m;
          if( ( NM % NTHREAD_M == 0 || im < NM ) &&
              is_elt_active )
          {
            if( ia_base == 0 ||
                NM*1 > NTHREAD_M*1 )
            {
              int iu_base = 0;
#pragma unroll
              for( iu_base=0; iu_base<NU; iu_base += NTHREAD_U )
              {
                const int iu = iu_base + sweeper_thread_u;

                if( NU % NTHREAD_U == 0 || iu < NU )
                {
                  *ref_vilocal( vilocal, sweeper->dims_b, NU, NTHREAD_M,
                                            sweeper_thread_m, iu ) =
                  *const_ref_state_flat( vi_this,
                                         sweeper->dims_b.ncell_x,
                                         sweeper->dims_b.ncell_y,
                                         sweeper->dims_b.ncell_z,
                                         sweeper->dims_b.ne,
                                         NM,
                                         NU,
                                         ix, iy, iz, ie, im, iu );
                  /*---Can use this for non-MIC case:
                    --- *const_ref_state( vi_this, sweeper->dims_b, NU,
                    ---                   ix, iy, iz, ie, im, iu );
                  ---*/
                }
              } /*---for iu---*/
            }
          }
        }
      }

      /*====================*/
      /*---Reassign threads from moments to angles---*/
      /*====================*/

      Sweeper_sync_amu_threads( sweeper );

      /*====================*/
      /*---Transform moments to angles---*/
      /*====================*/

      {
        /*
        WARNING!!!
        __assume( sweeper->dims_b.na % NTHREAD_A == 0 );
        */
        __assume_aligned( vslocal,  VEC_LEN * sizeof(P) );
        __assume_aligned( a_from_m, VEC_LEN * sizeof(P) );
        __assume_aligned( vilocal,  ( VEC_LEN < NTHREAD_M*NTHREAD_U ?
                                      VEC_LEN : NTHREAD_M*NTHREAD_U )
                                                                 * sizeof(P) );
#ifndef __CUDA_ARCH__
        int sweeper_thread_a = 0;
#pragma ivdep
#pragma simd assert, vectorlengthfor( P )
        for( sweeper_thread_a=0; sweeper_thread_a<NTHREAD_A;
                                                           ++sweeper_thread_a )
#endif
        {
          const int ia = ia_base + sweeper_thread_a;
          if( ia < sweeper->dims_b.na && is_elt_active )
          {
            int im_in_block = 0;
            int iu = 0;

            P v[NU];

#pragma unroll
            for( iu=0; iu<NU; ++iu )
            {
              v[iu] = ((P)0);
            }

            /*--------------------*/
            /*---Compute matvec in registers---*/
            /*--------------------*/

            for( im_in_block=0; im_in_block<NTHREAD_M; ++im_in_block )
            {
              const int im = im_base + im_in_block;

              if( NM % NTHREAD_M == 0 || im < NM )
              {
                const P a_from_m_this = *const_ref_a_from_m_flat(
                                             a_from_m,
                                             NM,
                                             sweeper->dims_b.na,
                                             im, ia, octant );
#pragma unroll
                for( iu=0; iu<NU; ++iu )
                {
                  v[iu] += a_from_m_this
                    * *const_ref_vilocal( vilocal, sweeper->dims_b,
                                        NU, NTHREAD_M, im_in_block, iu );
                }
              }
            } /*---for im_in_block---*/

            /*--------------------*/
            /*---Store/update to shared memory---*/
            /*--------------------*/

            if( im_base == 0 )
            {
#pragma unroll
              for( iu=0; iu<NU; ++iu )
              {
                vslocal[ ind_vslocal( sweeper->dims_b, NU, NTHREAD_A,
                                        sweeper_thread_a, iu ) ] = v[iu];
                /*---Can use this for non-MIC case
                *ref_vslocal( vslocal, sweeper->dims_b, NU, NTHREAD_A,
                                        sweeper_thread_a, iu )  = v[iu];
                */
              }
            }
            else
            {
#pragma unroll
              for( iu=0; iu<NU; ++iu )
              {
                vslocal[ ind_vslocal( sweeper->dims_b, NU, NTHREAD_A,
                                        sweeper_thread_a, iu ) ] += v[iu];
              }
            }
          } /*---if ia---*/
        }
      }
    } /*---for im_base---*/

    /*====================*/
    /*---Perform solve---*/
    /*====================*/

    /*
    WARNING!!!
    __assume( sweeper->dims_b.na % NTHREAD_A == 0 );
    */
    __assume_aligned( vslocal, VEC_LEN * sizeof(P) );
    __assume_aligned( facexy,  VEC_LEN * sizeof(P) );
    __assume_aligned( facexz,  VEC_LEN * sizeof(P) );
    __assume_aligned( faceyz,  VEC_LEN * sizeof(P) );
#ifndef __CUDA_ARCH__
    int sweeper_thread_a = 0;
#pragma ivdep
#pragma simd assert, vectorlengthfor( P )
    for( sweeper_thread_a=0; sweeper_thread_a<NTHREAD_A; ++sweeper_thread_a )
#endif
    {
      const int ia = ia_base + sweeper_thread_a;
      Quantities_solve( quan, vslocal,
                        ia, sweeper_thread_a, NTHREAD_A,
                        facexy, facexz, faceyz,
                        ix, iy, iz, ie,
                        ix+quan->ix_base, iy+quan->iy_base, iz+iz_base,
                        octant, octant_in_block,
                        sweeper->noctant_per_block,
                        sweeper->dims_b, sweeper->dims_g,
                        is_elt_active );
    }

    /*====================*/
    /*---Reassign threads from angles to moments---*/
    /*====================*/

    Sweeper_sync_amu_threads( sweeper );

    /*====================*/
    for( im_base=0; im_base<NM; im_base += NTHREAD_M )
    {
      {
        /*
        WARNING!!!
        __assume( sweeper->dims_b.na % NTHREAD_A == 0 );
        */
        __assume( sweeper->dims_b.nm == NM );
        __assume_aligned( vslocal,  VEC_LEN * sizeof(P) );
        __assume_aligned( m_from_a, VEC_LEN * sizeof(P) );
        __assume_aligned( vi_this,  ( VEC_LEN < NTHREAD_M*NTHREAD_U ?
                                      VEC_LEN : NTHREAD_M*NTHREAD_U )
                                                                 * sizeof(P) );
        __assume_aligned( vilocal,  ( VEC_LEN < NTHREAD_M*NTHREAD_U ?
                                      VEC_LEN : NTHREAD_M*NTHREAD_U )
                                                                 * sizeof(P) );
#ifndef __CUDA_ARCH__
        int sweeper_thread_u = 0;
        int sweeper_thread_m = 0;
        for( sweeper_thread_u=0; sweeper_thread_u<NTHREAD_U;
                                                           ++sweeper_thread_u )
#pragma ivdep
#pragma simd assert, vectorlengthfor( P )
        for( sweeper_thread_m=0; sweeper_thread_m<NTHREAD_M;
                                                           ++sweeper_thread_m )
#endif
        {
          const int im = im_base + sweeper_thread_m;

          P w[NU_PER_THREAD];

          int iu_per_thread = 0;
#pragma unroll
          for( iu_per_thread=0; iu_per_thread<NU_PER_THREAD; ++iu_per_thread )
          {
            w[iu_per_thread] = ((P)0);
          }

          /*====================*/
          /*---Transform angles to moments---*/
          /*====================*/

          if( ( NM % NTHREAD_M == 0 || im < NM ) &&
              is_elt_active )
          {
            int ia_in_block = 0;

            /*--------------------*/
            /*---Compute matvec in registers---*/
            /*--------------------*/

            /*---TODO: set up logic here to run fast for all cases---*/

#ifdef __MIC__
            if( ia_base + NTHREAD_A == sweeper->dims_b.na )
#else
            if( Bool_false )
#endif
            {
#ifdef __MIC__
/* "If applied to outer loop nests, the current implementation supports complete outer loop unrolling." */
#pragma unroll
#else
#pragma unroll 4
#endif
              for( ia_in_block=0; ia_in_block<NTHREAD_A; ++ia_in_block )
              {
                const int ia = ia_base + ia_in_block;

                const P m_from_a_this = m_from_a[
                                    ind_m_from_a_flat( sweeper->dims_b.nm,
                                                       sweeper->dims_b.na,
                                                       im, ia, octant ) ];
                {
#pragma unroll
                  for( iu_per_thread=0; iu_per_thread<NU_PER_THREAD;
                                                              ++iu_per_thread )
                  {
                    const int iu =  sweeper_thread_u + NTHREAD_U *
                                    iu_per_thread;

                    if( NU % NTHREAD_U == 0 || iu < NU )
                    {
                      w[ iu_per_thread ] +=
                          m_from_a_this
                          /*---Can use this for non-MIC case:
                            --- *const_ref_m_from_a( m_from_a, sweeper->dims_b,
                            ---                     im, ia, octant )
                          ---*/
                        * *const_ref_vslocal( vslocal, sweeper->dims_b, NU,
                                              NTHREAD_A, ia_in_block, iu );
                    }
                  } /*---for iu_per_thread---*/
                }
              } /*---for ia_in_block---*/
            }
            else /*---ia_base---*/
            {
#ifdef __MIC__
/* "If applied to outer loop nests, the current implementation supports complete outer loop unrolling." */
#pragma unroll
#else
#pragma unroll 4
#endif
              for( ia_in_block=0; ia_in_block<NTHREAD_A; ++ia_in_block )
              {
                const int ia = ia_base + ia_in_block;
                const Bool_t mask = ia < sweeper->dims_b.na;

                const P m_from_a_this = mask ? m_from_a[
                                    ind_m_from_a_flat( sweeper->dims_b.nm,
                                                       sweeper->dims_b.na,
                                                       im, ia, octant ) ]
                                    : ((P)0);
                {
#pragma unroll
                  for( iu_per_thread=0; iu_per_thread<NU_PER_THREAD;
                                                              ++iu_per_thread )
                  {
                    const int iu =  sweeper_thread_u + NTHREAD_U *
                                    iu_per_thread;

                    if( NU % NTHREAD_U == 0 || iu < NU )
                    {
                      w[ iu_per_thread ] += mask ?
                          m_from_a_this
                          /*---Can use this for non-MIC case:
                            --- *const_ref_m_from_a( m_from_a, sweeper->dims_b,
                            ---                     im, ia, octant )
                          ---*/
                        * *const_ref_vslocal( vslocal, sweeper->dims_b, NU,
                                              NTHREAD_A, ia_in_block, iu )
                        : ((P)0);
                    }
                  } /*---for iu_per_thread---*/
                }
              } /*---for ia_in_block---*/
            } /*---if ia_base---*/

            /*--------------------*/
            /*---Store/update to shared memory---*/
            /*--------------------*/

            if( ia_base == 0 ||
                NM*1 > NTHREAD_M*1 )
            {
#pragma unroll
              for( iu_per_thread=0; iu_per_thread<NU_PER_THREAD;
                                                              ++iu_per_thread )
              {
                const int iu =  sweeper_thread_u + NTHREAD_U *
                                iu_per_thread;

                if( NU % NTHREAD_U == 0 || iu < NU )
                {
                  *ref_volocal( volocal, sweeper->dims_b, NU, NTHREAD_M,
                            sweeper_thread_m, iu )  = w[ iu_per_thread ];
                }
              } /*---for iu_per_thread---*/
            }
            else
            {
#pragma unroll
              for( iu_per_thread=0; iu_per_thread<NU_PER_THREAD;
                                                              ++iu_per_thread )
              {
                const int iu =  sweeper_thread_u + NTHREAD_U *
                                iu_per_thread;

                if( (NU*1) % (NTHREAD_U*1) == 0 || iu < NU*1 )
                {
                  *ref_volocal( volocal, sweeper->dims_b, NU, NTHREAD_M,
                            sweeper_thread_m, iu ) += w[ iu_per_thread ];
                }
              } /*---for iu_per_thread---*/
            }
          } /*---if im---*/

          /*====================*/
          /*---Store/update portion of vo---*/
          /*====================*/

          if( ( (NM*1) % (NTHREAD_M*1) == 0 || im < NM*1 ) &&
              is_elt_active )
          {
            if( ia_base+NTHREAD_A >= sweeper->dims_b.na ||
                NM*1 > NTHREAD_M*1 )
            {
              int iu_base = 0;
#ifdef USE_OPENMP_VO_ATOMIC
#pragma unroll
              for( iu_base=0; iu_base<NU; iu += NTHREAD_U )
              {
                const int iu = iu_base + sweeper_thread_u;

                if( (NU*1) % (NTHREAD_U*1) == 0 || iu < (NU*1) )
                {
#pragma omp atomic update
                  *ref_state( vo_this, sweeper->dims_b, NU,
                              ix, iy, iz, ie, im, iu ) +=
                    *ref_volocal( volocal, sweeper->dims_b, NU, NTHREAD_M,
                                  sweeper_thread_m, iu );
                }
              }
#else /*---USE_OPENMP_VO_ATOMIC---*/
              if( ( ! do_block_init_this ) ||
                  ( NM*1 > NTHREAD_M*1 && ! ( ia_base==0 ) ) )
              {
#pragma unroll
                for( iu_base=0; iu_base<NU; iu_base += NTHREAD_U )
                {
                  const int iu = iu_base + sweeper_thread_u;

                  if( (NU*1) % (NTHREAD_U*1) == 0 || iu < NU*1 )
                  {
                    /*---Can use this for non-MIC case:
                      --- *ref_state( vo_this, sweeper->dims_b, NU,
                      ---             ix, iy, iz, ie, im, iu ) +=
                    */
                    *ref_state_flat( vo_this,
                                     sweeper->dims_b.ncell_x,
                                     sweeper->dims_b.ncell_y,
                                     sweeper->dims_b.ncell_z,
                                     sweeper->dims_b.ne,
                                     NM,
                                     NU,
                                     ix, iy, iz, ie, im, iu ) +=
                    *ref_volocal( volocal, sweeper->dims_b, NU, NTHREAD_M,
                                    sweeper_thread_m, iu );
                  }
                }
              }
              else
              {
#pragma unroll
                for( iu_base=0; iu_base<NU; iu_base += NTHREAD_U )
                {
                  const int iu = iu_base + sweeper_thread_u;

                  if( (NU*1) % (NTHREAD_U*1) == 0 || iu < NU*1 )
                  {
                    /*---Can use this for non-MIC case:
                      --- *ref_state( vo_this, sweeper->dims_b, NU,
                      ---             ix, iy, iz, ie, im, iu ) =
                    */
                    *ref_state_flat( vo_this,
                                     sweeper->dims_b.ncell_x,
                                     sweeper->dims_b.ncell_y,
                                     sweeper->dims_b.ncell_z,
                                     sweeper->dims_b.ne,
                                     NM,
                                     NU,
                                     ix, iy, iz, ie, im, iu )  =
                      *ref_volocal( volocal, sweeper->dims_b, NU, NTHREAD_M,
                                sweeper_thread_m, iu );
                  }
                }
              }
#endif /*---USE_OPENMP_VO_ATOMIC---*/
            }
          } /*---if im---*/
        }
      }

    } /*---for im_base---*/

  } /*---for ia_base---*/

}

/*===========================================================================*/
/*---Perform a sweep for a subblock---*/

TARGET_HD static inline void Sweeper_sweep_subblock(
  SweeperLite* __restrict__      sweeper,
  P* const __restrict__          vo_this,
  const P* const __restrict__    vi_this,
  P* const __restrict__          vilocal,
  P* const __restrict__          vslocal,
  P* const __restrict__          volocal,
  P* const __restrict__          facexy,
  P* const __restrict__          facexz,
  P* const __restrict__          faceyz,
  const P* const __restrict__    a_from_m,
  const P* const __restrict__    m_from_a,
  const Quantities* __restrict__ quan,
  const int                      octant,
  const int                      iz_base,
  const int                      octant_in_block,
  const int                      ixmin_subblock,
  const int                      ixmax_subblock,
  const int                      iymin_subblock,
  const int                      iymax_subblock,
  const int                      izmin_subblock,
  const int                      izmax_subblock,
  const Bool_t                   is_subblock_active,
  const int                      ixmin_semiblock,
  const int                      ixmax_semiblock,
  const int                      iymin_semiblock,
  const int                      iymax_semiblock,
  const int                      izmin_semiblock,
  const int                      izmax_semiblock,
  const int                      dir_x,
  const int                      dir_y,
  const int                      dir_z,
  const int                      dir_inc_x,
  const int                      dir_inc_y,
  const int                      dir_inc_z,
  const Bool_t                   do_block_init_this,
  const Bool_t                   is_octant_active )
{
  /*---Initializations---*/

  const int iemin = (   sweeper->dims.ne *
                      ( Sweeper_thread_e( sweeper )     ) )
                  /     sweeper->nthread_e;
  const int iemax = (   sweeper->dims.ne *
                      ( Sweeper_thread_e( sweeper ) + 1 ) )
                  /     sweeper->nthread_e;

  int ie = 0;

  const int ixbeg = dir_x==DIR_UP ? ixmin_subblock : ixmax_subblock;
  const int iybeg = dir_y==DIR_UP ? iymin_subblock : iymax_subblock;
  const int izbeg = dir_z==DIR_UP ? izmin_subblock : izmax_subblock;

  const int ixend = dir_x==DIR_DN ? ixmin_subblock : ixmax_subblock;
  const int iyend = dir_y==DIR_DN ? iymin_subblock : iymax_subblock;
  const int izend = dir_z==DIR_DN ? izmin_subblock : izmax_subblock;

  int ix = 0, iy = 0, iz = 0;

  /*--------------------*/
  /*---First perform any required boundary initializations---*/
  /*--------------------*/

  /*--------------------*/
  /*---Loop over energy groups owned by this energy thread---*/
  /*--------------------*/

  for( ie=iemin; ie<iemax; ++ie )
  {
    /*--------------------*/
    /*---Loop over cells in this subblock---*/
    /*--------------------*/

    for( iz=izbeg; iz!=izend+dir_inc_z; iz+=dir_inc_z )
    {
    for( iy=iybeg; iy!=iyend+dir_inc_y; iy+=dir_inc_y )
    {
    for( ix=ixbeg; ix!=ixend+dir_inc_x; ix+=dir_inc_x )
    {
      /*---Truncate loop region to block, semiblock and subblock---*/
      const Bool_t is_elt_active = ix <  sweeper->dims_b.ncell_x &&
                                   iy <  sweeper->dims_b.ncell_y &&
                                   iz <  sweeper->dims_b.ncell_z &&
                                   ix <= ixmax_semiblock &&
                                   iy <= iymax_semiblock &&
                                   iz <= izmax_semiblock &&
                                   is_subblock_active &&
                                   is_octant_active;
                                /* ix >= 0 &&
                                   iy >= 0 &&
                                   iz >= 0 &&
                                   ix >= ixmin_semiblock &&
                                   iy >= iymin_semiblock &&
                                   iz >= izmin_semiblock &&
                                   ix >= ixmin_subblock &&
                                   iy >= iymin_subblock &&
                                   iz >= izmin_subblock &&
                                   ix <= ixmax_subblock &&
                                   iy <= iymax_subblock &&
                                   iz <= izmax_subblock && (guaranteed) */

      if( is_elt_active )
      {
        /*--------------------*/
        /*---Set boundary condition if needed: xy---*/
        /*--------------------*/

        const int iz_g = iz + iz_base;
        if( ( iz_g == 0                         && dir_z == DIR_UP ) ||
            ( iz_g == sweeper->dims_g.ncell_z-1 && dir_z == DIR_DN ) )
        {
          const int ix_g = ix + quan->ix_base;
          const int iy_g = iy + quan->iy_base;
          /*---TODO: thread/vectorize in u, a---*/
          int iu = 0;
          for( iu=0; iu<NU; ++iu )
          {
            int ia = 0;
          for( ia=0; ia<sweeper->dims_b.na; ++ia )
          {
            *ref_facexy( facexy, sweeper->dims_b, NU,  
                         sweeper->noctant_per_block,
                         ix, iy, ie, ia, iu, octant_in_block )     
               = Quantities_init_facexy( quan, ix_g, iy_g, iz_g-dir_inc_z,
                                         ie, ia, iu, octant, sweeper->dims_g );
          }
          }
        }

        /*--------------------*/
        /*---Set boundary condition if needed: xz---*/
        /*--------------------*/

        const int iy_g = iy + quan->iy_base;
        if( ( iy_g == 0                         && dir_y == DIR_UP ) ||
            ( iy_g == sweeper->dims_g.ncell_y-1 && dir_y == DIR_DN ) )
        {
          const int ix_g = ix + quan->ix_base;
          const int iz_g = iz +       iz_base;
          /*---TODO: thread/vectorize in u, a---*/
          int iu = 0;
          for( iu=0; iu<NU; ++iu )
          {
            int ia = 0;
          for( ia=0; ia<sweeper->dims_b.na; ++ia )
          {
            *ref_facexz( facexz, sweeper->dims_b, NU,  
                         sweeper->noctant_per_block,
                         ix, iz, ie, ia, iu, octant_in_block )     
               = Quantities_init_facexz( quan, ix_g, iy_g-dir_inc_y, iz_g,
                                         ie, ia, iu, octant, sweeper->dims_g );
          }
          }
        }

        /*--------------------*/
        /*---Set boundary condition if needed: yz---*/
        /*--------------------*/

        const int ix_g = ix + quan->ix_base;
        if( ( ix_g == 0                         && dir_x == DIR_UP ) ||
            ( ix_g == sweeper->dims_g.ncell_x-1 && dir_x == DIR_DN ) )
        {
          const int iy_g = iy + quan->iy_base;
          const int iz_g = iz +       iz_base;
          /*---TODO: thread/vectorize in u, a---*/
          int iu = 0;
          for( iu=0; iu<NU; ++iu )
          {
            int ia = 0;
          for( ia=0; ia<sweeper->dims_b.na; ++ia )
          {
            *ref_faceyz( faceyz, sweeper->dims_b, NU,  
                         sweeper->noctant_per_block,
                         iy, iz, ie, ia, iu, octant_in_block )     
               = Quantities_init_faceyz( quan, ix_g-dir_inc_x, iy_g, iz_g,
                                         ie, ia, iu, octant, sweeper->dims_g );
          }
          }
        }

      } /*---is_elt_active---*/

    }
    }
    } /*---ix/iy/iz---*/

  } /*---ie---*/

  /*--------------------*/
  /*---Now perform actual sweep--*/
  /*--------------------*/

  /*--------------------*/
  /*---Loop over energy groups owned by this energy thread---*/
  /*--------------------*/

  for( ie=iemin; ie<iemax; ++ie )
  {
    /*--------------------*/
    /*---Sweep subblock: loop over cells, in proper direction---*/
    /*--------------------*/

    for( iz=izbeg; iz!=izend+dir_inc_z; iz+=dir_inc_z )
    {
    for( iy=iybeg; iy!=iyend+dir_inc_y; iy+=dir_inc_y )
    {
    for( ix=ixbeg; ix!=ixend+dir_inc_x; ix+=dir_inc_x )
    {
      /*---Truncate loop region to block, semiblock and subblock---*/
      const Bool_t is_elt_active = ix <  sweeper->dims_b.ncell_x &&
                                   iy <  sweeper->dims_b.ncell_y &&
                                   iz <  sweeper->dims_b.ncell_z &&
                                   ix <= ixmax_semiblock &&
                                   iy <= iymax_semiblock &&
                                   iz <= izmax_semiblock &&
                                   is_subblock_active &&
                                   is_octant_active;
                                /* ix >= 0 &&
                                   iy >= 0 &&
                                   iz >= 0 &&
                                   ix >= ixmin_semiblock &&
                                   iy >= iymin_semiblock &&
                                   iz >= izmin_semiblock &&
                                   ix >= ixmin_subblock &&
                                   iy >= iymin_subblock &&
                                   iz >= izmin_subblock &&
                                   ix <= ixmax_subblock &&
                                   iy <= iymax_subblock &&
                                   iz <= izmax_subblock && (guaranteed) */

      /*--------------------*/
      /*---Perform sweep on cell---*/
      /*--------------------*/
      Sweeper_sweep_cell( sweeper, vo_this, vi_this, vilocal, vslocal, volocal,
                          facexy, facexz, faceyz, a_from_m, m_from_a, quan,
                          octant, iz_base, octant_in_block, ie, ix, iy, iz,
                          do_block_init_this,
                          is_elt_active );
    }
    }
    } /*---ix/iy/iz---*/
  } /*---ie---*/
}

/*===========================================================================*/
/*---Perform a sweep for a semiblock---*/

TARGET_HD static inline void Sweeper_sweep_semiblock(
  SweeperLite*           sweeper,
  P* __restrict__        vo_this,
  const P* __restrict__  vi_this,
  P* __restrict__        facexy,
  P* __restrict__        facexz,
  P* __restrict__        faceyz,
  const P* __restrict__  a_from_m,
  const P* __restrict__  m_from_a,
  const Quantities*      quan,
  const StepInfo         stepinfo,
  const int              octant_in_block,
  const int              ixmin_semiblock,
  const int              ixmax_semiblock,
  const int              iymin_semiblock,
  const int              iymax_semiblock,
  const int              izmin_semiblock,
  const int              izmax_semiblock,
  const Bool_t           do_block_init_this,
  const Bool_t           is_octant_active )
{
  /*---Calculate needed quantities---*/

  const int octant  = stepinfo.octant;
  const int iz_base = stepinfo.block_z * sweeper->dims_b.ncell_z;

  P* __restrict__ vilocal = Sweeper_vilocal_this_( sweeper );
  P* __restrict__ vslocal = Sweeper_vslocal_this_( sweeper );
  P* __restrict__ volocal = Sweeper_volocal_this_( sweeper );

  const int dir_x = Dir_x( octant );
  const int dir_y = Dir_y( octant );
  const int dir_z = Dir_z( octant );

  const int dir_inc_x = Dir_inc(dir_x);
  const int dir_inc_y = Dir_inc(dir_y);
  const int dir_inc_z = Dir_inc(dir_z);

  /*---Number of subblocks---*/

  const int nsubblock_x = iceil( ixmax_semiblock-ixmin_semiblock+1,
                                               sweeper->ncell_x_per_subblock );
  const int nsubblock_y = iceil( iymax_semiblock-iymin_semiblock+1,
                                               sweeper->ncell_y_per_subblock );
  const int nsubblock_z = iceil( izmax_semiblock-izmin_semiblock+1,
                                               sweeper->ncell_z_per_subblock );

  if( IS_USING_OPENMP_TASKS )
  {
    /*--------------------*/
    /*---CASE: Tasking---*/
    /*--------------------*/

    /*---Determine which subblock to sweep---*/

    const int subblock_x = dir_x == DIR_UP ?
                                             Sweeper_thread_x( sweeper ) :
                           nsubblock_x - 1 - Sweeper_thread_x( sweeper );

    const int subblock_y = dir_y == DIR_UP ?
                                             Sweeper_thread_y( sweeper ) :
                           nsubblock_y - 1 - Sweeper_thread_y( sweeper );

    const int subblock_z = dir_z == DIR_UP ?
                                             Sweeper_thread_z( sweeper ) :
                           nsubblock_z - 1 - Sweeper_thread_z( sweeper );

    const Bool_t is_subblock_active = Bool_true;

    /*---Compute subblock bounds, inclusive of endpoints---*/

    const int ixmin_subblock = ixmin_semiblock +
                            sweeper->ncell_x_per_subblock *  subblock_x;
    const int ixmax_subblock = ixmin_semiblock +
                            sweeper->ncell_x_per_subblock * (subblock_x+1) - 1;
    const int iymin_subblock = iymin_semiblock +
                            sweeper->ncell_y_per_subblock *  subblock_y;
    const int iymax_subblock = iymin_semiblock +
                            sweeper->ncell_y_per_subblock * (subblock_y+1) - 1;
    const int izmin_subblock = izmin_semiblock +
                            sweeper->ncell_z_per_subblock *  subblock_z;
    const int izmax_subblock = izmin_semiblock +
                            sweeper->ncell_z_per_subblock * (subblock_z+1) - 1;

    /*--------------------*/
    /*---Perform sweep on subblock---*/
    /*--------------------*/

    Sweeper_sweep_subblock( sweeper, vo_this, vi_this,
                            vilocal, vslocal, volocal,
                            facexy, facexz, faceyz, a_from_m, m_from_a, quan,
                            octant, iz_base, octant_in_block,
                            ixmin_subblock, ixmax_subblock,
                            iymin_subblock, iymax_subblock,
                            izmin_subblock, izmax_subblock,
                            is_subblock_active,
                            ixmin_semiblock, ixmax_semiblock,
                            iymin_semiblock, iymax_semiblock,
                            izmin_semiblock, izmax_semiblock,
                            dir_x, dir_y, dir_z,
                            dir_inc_x, dir_inc_y, dir_inc_z,
                            do_block_init_this,
                            is_octant_active );
  }
  else /*---if tasking---*/
  {
    /*--------------------*/
    /*---CASE: Threading---*/
    /*--------------------*/

    /*---Size of chunk, measured in subblocks---*/

    const int nsubblock_x_per_chunk = nsubblock_x;
    const int nsubblock_y_per_chunk = sweeper->nthread_y;
    const int nsubblock_z_per_chunk = sweeper->nthread_z;

    /*---Number of chunks, rounded up as needed---*/

    const int nchunk_x = iceil( nsubblock_x, nsubblock_x_per_chunk );
    const int nchunk_y = iceil( nsubblock_y, nsubblock_y_per_chunk );
    const int nchunk_z = iceil( nsubblock_z, nsubblock_z_per_chunk );

    /*---Upsize to ensure dependencies satisfied---*/

    const int nsubblock_x_per_chunk_up
        = imax( nsubblock_x_per_chunk,
          imax( nsubblock_y_per_chunk,
               iceil( nsubblock_z_per_chunk, nchunk_y ) ) );

    /*---Calculate info for stacked domain---*/

    const int nchunk_yz = nchunk_y * nchunk_z;

    const int nsubblock_x_stacked = nsubblock_x_per_chunk_up * nchunk_yz;

    const int nsubblockwave = nsubblock_x_stacked
                            + nsubblock_y_per_chunk
                            + nsubblock_z_per_chunk - 2;

    int subblockwave = 0;

    /*--------------------*/
    /*---Loop over subblock wavefronts---*/
    /*--------------------*/

    for( subblockwave=0; subblockwave<nsubblockwave; ++subblockwave )
    {
      /*---Get coordinates of subblock in stacked domain---*/

      const int subblock_y_stacked = Sweeper_thread_y( sweeper );
      const int subblock_z_stacked = Sweeper_thread_z( sweeper );
      const int subblock_x_stacked = subblockwave
                                   - subblock_y_stacked
                                   - subblock_z_stacked;

      /*---x subblock coordinate in unstacked domain---*/

      const int subblock_x = ( dir_x==DIR_UP ?  subblock_x_stacked :
                         (nsubblock_x_stacked-1-subblock_x_stacked) )
                    % nsubblock_x_per_chunk_up;

      /*---y, z chunk coordinates in unstacked domain---*/

      const int chunk_yz_stacked = subblock_x_stacked
                                 / nsubblock_x_per_chunk_up;

      const int chunk_z = dir_z==DIR_UP ? chunk_yz_stacked  / nchunk_y :
                             (nchunk_yz-1-chunk_yz_stacked) / nchunk_y;

      const int chunk_y = dir_y==DIR_UP ? chunk_yz_stacked  % nchunk_y :
                             (nchunk_yz-1-chunk_yz_stacked) % nchunk_y;

      /*---y, z subblock coordinate in unstacked domain---*/

      const int subblock_y = chunk_y * nsubblock_y_per_chunk +
                          (dir_y==DIR_UP ? subblock_y_stacked :
               nsubblock_y_per_chunk - 1 - subblock_y_stacked);

      const int subblock_z = chunk_z * nsubblock_z_per_chunk +
                          (dir_z==DIR_UP ? subblock_z_stacked :
               nsubblock_z_per_chunk - 1 - subblock_z_stacked);

      const Bool_t is_subblock_active =
          subblock_x_stacked >= 0 && subblock_x_stacked < nsubblock_x_stacked &&
          subblock_z         >= 0 && subblock_z         < nsubblock_z &&
          subblock_y         >= 0 && subblock_y         < nsubblock_y &&
          subblock_x         >= 0 && subblock_x         < nsubblock_x;

      /*---Compute subblock bounds, inclusive of endpoints---*/

      const int ixmin_subblock = ixmin_semiblock +
                            sweeper->ncell_x_per_subblock *  subblock_x;
      const int ixmax_subblock = ixmin_semiblock +
                            sweeper->ncell_x_per_subblock * (subblock_x+1) - 1;
      const int iymin_subblock = iymin_semiblock +
                            sweeper->ncell_y_per_subblock *  subblock_y;
      const int iymax_subblock = iymin_semiblock +
                            sweeper->ncell_y_per_subblock * (subblock_y+1) - 1;
      const int izmin_subblock = izmin_semiblock +
                            sweeper->ncell_z_per_subblock *  subblock_z;
      const int izmax_subblock = izmin_semiblock +
                            sweeper->ncell_z_per_subblock * (subblock_z+1) - 1;

      /*--------------------*/
      /*---Perform sweep on subblock---*/
      /*--------------------*/
      Sweeper_sweep_subblock( sweeper, vo_this, vi_this,
                              vilocal, vslocal, volocal,
                              facexy, facexz, faceyz, a_from_m, m_from_a, quan,
                              octant, iz_base, octant_in_block,
                              ixmin_subblock, ixmax_subblock,
                              iymin_subblock, iymax_subblock,
                              izmin_subblock, izmax_subblock,
                              is_subblock_active,
                              ixmin_semiblock, ixmax_semiblock,
                              iymin_semiblock, iymax_semiblock,
                              izmin_semiblock, izmax_semiblock,
                              dir_x, dir_y, dir_z,
                              dir_inc_x, dir_inc_y, dir_inc_z,
                              do_block_init_this,
                              is_octant_active );

      if( subblockwave != nsubblockwave-1 )
      {
        Sweeper_sync_yz_threads( sweeper );
      }
    } /*---subblockwave---*/
  } /*---if tasking---*/
}

/*===========================================================================*/
/*---Helper function to calculate semiblock bounds---*/

TARGET_HD static inline void Sweeper_get_semiblock_bounds(
  Bool_t* __restrict__ has_lo,
  Bool_t* __restrict__ has_hi,
  int*    __restrict__ imin,
  int*    __restrict__ imax,
  int*    __restrict__ imax_up2,
  int ncell,
  int dim,
  int dir,
  int semiblock_step,
  int nsemiblock
  )
{
  /*===================================================================
  =    is_semiblocked: indicate whether the block is broken into
  =      semiblocks along the x axis.
  =    is_semiblock_lo: on this semiblock step for this thread
  =      do we process the lower or the higher semiblock along
  =      the x axis.  Only meaningful if is_semiblocked.
  =    has_lo: does this semiblock contain the lowest cell of the
  =      block along the selected axis.
  =    has_hi: similarly.
  =    imin: the lowest cell boundary of the semiblock within the
  =      block along the selected axis, inclusive of endpoints.
  =    imax: similarly.
  =    imax_up2: make lower/upper semiblock of same size, so that
  =      every device octant thread has the same loop trip count,
  =      so that syncthreads works right, later mask out the
  =      added cells so no real work is done.
  ===================================================================*/

  const Bool_t is_semiblocked = is_axis_semiblocked(nsemiblock, dim);
  const Bool_t is_semiblock_lo = is_semiblock_min_when_semiblocked(
            nsemiblock, semiblock_step, dim, dir );

  *has_lo =     is_semiblock_lo   || ! is_semiblocked;
  *has_hi = ( ! is_semiblock_lo ) || ! is_semiblocked;

  /*---Get semiblock bounds, inclusive of endpoints---*/

  *imin =     *has_lo   ? 0 : ( (ncell+1) / 2 );
  *imax = ( ! *has_hi ) ?     ( (ncell+1) / 2 - 1 ) : ( ncell - 1 );

  *imax_up2 = ( is_semiblocked &&
                ( ncell % 2 ) &&
                ( ! is_semiblock_lo ) )
              ?  ( *imax + 1 ) : *imax;
}                  

/*===========================================================================*/
/*---Perform a sweep for a block, implementation---*/

TARGET_HD void Sweeper_sweep_block_impl(
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
  unsigned long int      do_block_init )
{
  /*---Declarations---*/
    const int noctant_per_block = sweeper.noctant_per_block;

    const int nsemiblock = sweeper.nsemiblock;

    int semiblock_step = 0;

    /*=========================================================================
    =    OpenMP-parallelizing octants leads to the problem that for the same
    =    step, two octants may be updating the same location in a state vector.
    =
    =    One solution is to make the state vector update atomic, which is
    =    likely to be inefficient depending on the system.
    =
    =    The alternative used here is to break the step into sub steps
    =    and break the block into subregions such that during a sub-step,
    =    different octants in different threads update disjoint subregions.
    =
    =    First, note that octants are assigned to threads as follows:
    =      nthread_octant==1: one thread for all octants.
    =      nthread_octant==2: -x and +x octants assigned to different threads.
    =      nthread_octant==4: -y and +y octants also have different threads.
    =      nthread_octant==8: -z and +z octants also have different threads.
    =
    =    Along each coordinate axis for which two threads are assigned,
    =    the block is divided into two halves.  This gives a set of semiblocks.
    =
    =    The semiblocks are visited by the semiblock loop in an ordering
    =    which is lexicographical, either forward or reverse direction
    =    depending on the direction specified by that octant along the axis.
    =    This is set up so that (1) the disjointness condition described
    =    above holds, and (2) the cells are visited in an order that
    =    satisfies the sweep recursion.
    =
    =    NOTES:
    =    - For the unthreaded case, nsemiblock and noctant_per_block
    =      can be set to any of the allowed values and the algorithm will
    =      work properly.
    =    - If nsemiblock==noctant_per_block, then any value of nthread_octant
    =      applied to the OpenMP loop will work ok.
    =    - If nsemiblock<noctant_per_block==nthread_octant, then
    =      a potential race condition will occur.  This can be fixed by
    =      making the update of vo at the end of Sweeper_sweep_semiblock atomic.
    =      What is in question here is the overhead of the semiblock loop.
    =      One might want to reduce the number of semiblocks while keeping
    =      noctant_per_block==nthread_octant high to get more thread
    =      parallelism but possibly not too high so as to control the
    =      wavefront latency.
    =========================================================================*/

    /*--------------------*/
    /*---Loop over semiblock steps---*/
    /*--------------------*/

    for( semiblock_step=0; semiblock_step<nsemiblock; ++semiblock_step )
    {
#ifdef USE_OPENMP_TASKS
      /*--------------------*/
      /*---Enter parallel region where tasks will be launched---*/
      /*--------------------*/

      int thread_octant = 0;
      int thread_e = 0;

#pragma omp parallel for firstprivate(sweeper) collapse(2)
      for( thread_octant=0; thread_octant<sweeper.nthread_octant;
                                                             ++thread_octant)
      {
      for( thread_e=0; thread_e<sweeper.nthread_e; ++thread_e)
      {
        sweeper.thread_octant = thread_octant;
        sweeper.thread_e = thread_e;

      /*--------------------*/
      /*---Loop over the space of all tasks to be launched---*/
      /*--------------------*/

      /*---Here we assign one thread/task per subblock---*/

      int thread_z = 0;
      for( thread_z=0; thread_z<sweeper.nthread_z; ++thread_z)
      {
        sweeper.thread_z = thread_z;

      int thread_y = 0;
      for( thread_y=0; thread_y<sweeper.nthread_y; ++thread_y)
      {
        sweeper.thread_y = thread_y;

      int thread_x = 0;
      for( thread_x=0; thread_x<sweeper.nthread_x; ++thread_x)
      {
        sweeper.thread_x = thread_x;

      /*--------------------*/
      /*---Determine dependencies---*/
      /*--------------------*/

      const char* __restrict__ dep_in_x = Sweeper_task_dependency( &sweeper,
                thread_x-1, thread_y,   thread_z,   thread_e, thread_octant );

      const char* __restrict__ dep_in_y = Sweeper_task_dependency( &sweeper,
                thread_x,   thread_y-1, thread_z,   thread_e, thread_octant );

      const char* __restrict__ dep_in_z = Sweeper_task_dependency( &sweeper,
                thread_x,   thread_y,   thread_z-1, thread_e, thread_octant );

      const char* __restrict__ dep_out = Sweeper_task_dependency( &sweeper,
                thread_x,   thread_y,   thread_z,   thread_e, thread_octant );

      /*
        printf("%i %i "
               "Submitting task %i %i %i %i %i  thread %i numthreads %i\n",
               step, semiblock_step,
               thread_x, thread_y, thread_z, thread_e, thread_octant,
               omp_get_thread_num(), omp_get_num_threads());
      */

      /*--------------------*/
      /*---Launch this task---*/
      /*--------------------*/

#pragma omp task \
      depend(in:  dep_in_x[0]) \
      depend(in:  dep_in_y[0]) \
      depend(in:  dep_in_z[0]) \
      depend(out: dep_out[0])
      {
      /*
        printf("%i %i                   "
               "Commencing task %i %i %i %i %i  thread %i\n",
               step, semiblock_step,
               thread_x, thread_y, thread_z, thread_e, thread_octant,
               omp_get_thread_num());
      */
#endif

      /*--------------------*/
      /*---Loop over octants in octant block---*/
      /*---That is, octants that are computed for this semiblock step---*/
      /*--------------------*/

        const int octant_in_block_min =
                             (   sweeper.noctant_per_block *
                               ( Sweeper_thread_octant( &sweeper )     ) )
                           /     sweeper.nthread_octant;
        const int octant_in_block_max =
                             (   sweeper.noctant_per_block *
                               ( Sweeper_thread_octant( &sweeper ) + 1 ) )
                           /     sweeper.nthread_octant;

        int octant_in_block = 0;

      for( octant_in_block=octant_in_block_min;
           octant_in_block<octant_in_block_max; ++octant_in_block )
      {
        /*---Get step info---*/

        const StepInfo stepinfo = stepinfoall.stepinfo[octant_in_block];

        const Bool_t is_octant_active = stepinfo.is_active;

        const int dir_x = Dir_x( stepinfo.octant );
        const int dir_y = Dir_y( stepinfo.octant );
        const int dir_z = Dir_z( stepinfo.octant );

        /*--------------------*/
        /*---Compute semiblock bounds---*/
        /*--------------------*/

        Bool_t is_semiblock_min_x = 0, is_semiblock_max_x = 0;
        int ixmin_semiblock = 0, ixmax_semiblock = 0, ixmax_semiblock_up2 = 0;

        Sweeper_get_semiblock_bounds(&is_semiblock_min_x, &is_semiblock_max_x,
          &ixmin_semiblock, &ixmax_semiblock, &ixmax_semiblock_up2,
          sweeper.dims_b.ncell_x, DIM_X, dir_x, semiblock_step, nsemiblock);

        /*--------------------*/

        Bool_t is_semiblock_min_y = 0, is_semiblock_max_y = 0;
        int iymin_semiblock = 0, iymax_semiblock = 0, iymax_semiblock_up2 = 0;

        Sweeper_get_semiblock_bounds(&is_semiblock_min_y, &is_semiblock_max_y,
          &iymin_semiblock, &iymax_semiblock, &iymax_semiblock_up2,
          sweeper.dims_b.ncell_y, DIM_Y, dir_y, semiblock_step, nsemiblock);

        /*--------------------*/

        Bool_t is_semiblock_min_z = 0, is_semiblock_max_z = 0;
        int izmin_semiblock = 0, izmax_semiblock = 0, izmax_semiblock_up2 = 0;

        Sweeper_get_semiblock_bounds(&is_semiblock_min_z, &is_semiblock_max_z,
          &izmin_semiblock, &izmax_semiblock, &izmax_semiblock_up2,
          sweeper.dims_b.ncell_z, DIM_Z, dir_z, semiblock_step, nsemiblock);

        /*--------------------*/
        /*---Perform sweep over subblocks in semiblock---*/
        /*---(for tasking case, this task sweeps one subblock in semiblock---*/
        /*--------------------*/

        const int iz_base = stepinfo.block_z * sweeper.dims_b.ncell_z;

        const P* vi_this = const_ref_state( vi, sweeper.dims, NU, 0, 0,
                                                            iz_base, 0, 0, 0 );
        P* vo_this =             ref_state( vo, sweeper.dims, NU, 0, 0,
                                                            iz_base, 0, 0, 0 );

        const int do_block_init_this = !! ( do_block_init &
                         ( ((unsigned long int)1) <<
                           ( octant_in_block + noctant_per_block *
                             semiblock_step ) ) );

        Sweeper_sweep_semiblock( &sweeper, vo_this, vi_this,
                                 facexy, facexz, faceyz,
                                 a_from_m, m_from_a,
                                 &quan, stepinfo, octant_in_block,
                                 ixmin_semiblock, ixmax_semiblock_up2,
                                 iymin_semiblock, iymax_semiblock_up2,
                                 izmin_semiblock, izmax_semiblock_up2,
                                 do_block_init_this,
                                 is_octant_active );

      } /*---octant_in_block---*/

#ifdef USE_OPENMP_TASKS
      /*
        printf("%i %i                                                 "
               "Completing task %i %i %i %i %i  thread %i\n",
               step, semiblock_step,
               thread_x, thread_y, thread_z, thread_e, thread_octant,
               omp_get_thread_num());
      */

      } /*---omp task---*/
      }
      }
      }
      }
      } /*---omp parallel for---*/ /*---NOTE: implicit sync here---*/
#else
      /*---Sync between semiblock steps---*/
      Sweeper_sync_octant_threads( &sweeper );
#endif

    } /*---semiblock---*/
}

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
  unsigned long int      do_block_init )
{
    Sweeper_sweep_block_impl( sweeper, vo, vi, facexy, facexz, faceyz,
                              a_from_m, m_from_a, step, quan,
                              proc_x_min, proc_x_max, proc_y_min, proc_y_max,
                              stepinfoall, do_block_init );
}

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_sweeper_kba_c_kernels_h_---*/

/*---------------------------------------------------------------------------*/
