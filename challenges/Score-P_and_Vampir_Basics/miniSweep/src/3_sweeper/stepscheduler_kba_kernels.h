/*---------------------------------------------------------------------------*/
/*!
 * \file   stepscheduler_kba_kernels.h
 * \author Wayne Joubert
 * \date   Tue Jan 28 16:37:41 EST 2014
 * \brief  stepscheduler_kba, code for comp. kernel.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _stepscheduler_kba_kernels_h_
#define _stepscheduler_kba_kernels_h_

#include "types_kernels.h"
#include "definitions_kernels.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Struct with info describing a sweep step---*/

typedef struct
{
  int     block_z;
  int     octant;
  Bool_t  is_active;
} StepInfo;

/*===========================================================================*/
/*---8 copies of the same---*/

typedef struct
{
  StepInfo stepinfo[NOCTANT];
} StepInfoAll;

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_stepscheduler_kba_kernels_h_---*/

/*---------------------------------------------------------------------------*/
