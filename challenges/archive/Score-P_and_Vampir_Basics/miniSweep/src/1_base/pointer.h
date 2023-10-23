/*---------------------------------------------------------------------------*/
/*!
 * \file   pointer.h
 * \author Wayne Joubert
 * \date   Tue Apr 22 14:57:52 EDT 2014
 * \brief  Pseudo-class for mirror host/device arrays, header.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _pointer_h_
#define _pointer_h_

#include <stddef.h>

#include "types.h"
#include "env.h"
#include "pointer_kernels.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Null object---*/
  
Pointer Pointer_null(void);
  
/*===========================================================================*/
/*---Conditional host or device pointer---*/

static P* __restrict__ Pointer_active( Pointer* p )
{
  Assert( p );
  return p->is_using_device_ ? p->d_ : p->h_;
}

/*===========================================================================*/
/*---Conditional host or device pointer---*/

static P* __restrict__ Pointer_const_active( const Pointer* p )
{
  Assert( p );
  return p->is_using_device_ ? p->d_ : p->h_;
}

/*===========================================================================*/
/*---Pseudo-constructors---*/

void Pointer_create( Pointer* p,
                     size_t   n,
                     Bool_t   is_using_device );

/*---------------------------------------------------------------------------*/

void Pointer_create_alias( Pointer* p,
                           Pointer* source,
                           size_t   base,
                           size_t   n );

/*---------------------------------------------------------------------------*/

void Pointer_set_pinned( Pointer* p,
                         Bool_t   is_pinned );

/*===========================================================================*/
/*---Pseudo-destructor---*/

void Pointer_destroy( Pointer* p );

/*===========================================================================*/
/*---De/allocate memory---*/

void Pointer_allocate_h_( Pointer* p );

/*---------------------------------------------------------------------------*/

void Pointer_allocate_d_( Pointer* p );

/*---------------------------------------------------------------------------*/

void Pointer_allocate( Pointer* p );

/*---------------------------------------------------------------------------*/

void Pointer_deallocate_h_( Pointer* p );

/*---------------------------------------------------------------------------*/

void Pointer_deallocate_d_( Pointer* p );

/*---------------------------------------------------------------------------*/

void Pointer_deallocate( Pointer* p );

/*===========================================================================*/
/*---Copy memory---*/

void Pointer_update_h( Pointer* p );

/*---------------------------------------------------------------------------*/

void Pointer_update_d( Pointer* p );

/*---------------------------------------------------------------------------*/

void Pointer_update_h_stream( Pointer* p, Stream_t stream );

/*---------------------------------------------------------------------------*/

void Pointer_update_d_stream( Pointer* p, Stream_t stream );

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_pointer_h_---*/

/*---------------------------------------------------------------------------*/
