/*---------------------------------------------------------------------------*/
/*!
 * \file   types_kernels.h
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Declarations for basic types, code for comp. kernel.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _types_kernels_h_
#define _types_kernels_h_

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Function attributes---*/

#ifdef USE_CUDA

#define TARGET_G  __global__
#define TARGET_HD __host__ __device__

#else

#define TARGET_G
#define TARGET_HD

#endif

/*===========================================================================*/
/*---Basic types---*/

/*---Boolean type---*/

typedef int Bool_t;

enum{ Bool_true = (1==1), Bool_false = (0==1) };

/*---Default floating point type---*/

typedef double P;
enum{ P_IS_DOUBLE = Bool_true };

TARGET_HD static inline P P_zero() { return (P)0; }
TARGET_HD static inline P P_one()  { return (P)1; }

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_types_kernels_h_---*/

/*---------------------------------------------------------------------------*/
