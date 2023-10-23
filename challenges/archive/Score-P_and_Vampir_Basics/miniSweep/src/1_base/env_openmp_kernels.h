/*---------------------------------------------------------------------------*/
/*!
 * \file   env_openmp_kernels.h
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Environment settings for openmp, code for comp. kernel.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _env_openmp_kernels_h_
#define _env_openmp_kernels_h_

#ifdef USE_OPENMP
#include "omp.h"
#endif

#include "types_kernels.h"
#include "env_assert_kernels.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Enums---*/

#ifdef USE_OPENMP
enum{ IS_USING_OPENMP = Bool_true };
#else
enum{ IS_USING_OPENMP = Bool_false };
#endif 

/*===========================================================================*/
/*---Get openmp current thread number---*/

TARGET_HD static inline int Env_omp_thread()
{
  int result = 0;
#ifdef USE_OPENMP
  result = omp_get_thread_num();  
#endif
  return result;
}

/*===========================================================================*/
/*---Are we in an openmp threaded region---*/

TARGET_HD static inline Bool_t Env_omp_in_parallel()
{
  Bool_t result = Bool_false;
#ifdef USE_OPENMP
  result = omp_in_parallel();
#endif
  return result;
}

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_env_openmp_kernels_h_---*/

/*---------------------------------------------------------------------------*/
