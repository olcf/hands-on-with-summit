/*---------------------------------------------------------------------------*/
/*!
 * \file   pointer_kernels.h
 * \author Wayne Joubert
 * \date   Tue Apr 22 14:57:52 EDT 2014
 * \brief  Pseudo-class for mirror host/device arrays, code for comp. kernel.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _pointer_kernels_h_
#define _pointer_kernels_h_

#include <stddef.h>

#include "types_kernels.h"
#include "env_assert_kernels.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Enums---*/

enum{ IS_USING_DEVICE = Bool_true, IS_NOT_USING_DEVICE = Bool_false };
enum{ IS_PINNED = Bool_true, IS_NOT_PINNED = Bool_false };

/*===========================================================================*/
/*---Pointer struct---*/

typedef struct
{
  size_t          n_;
  P* __restrict__ h_;
  P* __restrict__ d_;
  Bool_t          is_using_device_;
  Bool_t          is_pinned_;
  Bool_t          is_alias_;
} Pointer;

/*===========================================================================*/
/*---Accessors---*/

TARGET_HD static inline P* __restrict__ Pointer_h( Pointer* p )
{
  Assert( p );
  Assert( p->h_ );
  return p->h_;
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline const P* __restrict__ Pointer_const_h( const Pointer* p )
{
  Assert( p );
  Assert( p->h_ );
  return p->h_;
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline P* __restrict__ Pointer_d( Pointer* p )
{
  Assert( p );
  Assert( p->d_ );
  Assert( p->is_using_device_ );
  return p->d_;
}

/*---------------------------------------------------------------------------*/

TARGET_HD static inline const P* __restrict__ Pointer_const_d( const Pointer* p )
{
  Assert( p );
  Assert( p->d_ );
  Assert( p->is_using_device_ );
  return p->d_;
}

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_pointer_kernels_h_---*/

/*---------------------------------------------------------------------------*/
