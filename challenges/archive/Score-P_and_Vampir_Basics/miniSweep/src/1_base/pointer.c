/*---------------------------------------------------------------------------*/
/*!
 * \file   pointer.c
 * \author Wayne Joubert
 * \date   Tue Apr 22 14:57:52 EDT 2014
 * \brief  Pseudo-class for mirror host/device arrays.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#include <stddef.h>
#include <string.h>

#include "types.h"
#include "env.h"
#include "pointer.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Null object---*/

Pointer Pointer_null()
{
  Pointer result;
  memset( (void*)&result, 0, sizeof(Pointer) );
  return result;
}

/*===========================================================================*/
/*---Pseudo-constructors etc.---*/

void Pointer_create( Pointer* p,
                     size_t   n,
                     Bool_t   is_using_device )
{
  Assert( p );
  Assert( n+1 >= 1 );

  p->h_ = NULL;
  p->d_ = NULL;
  p->n_ = n;
  p->is_using_device_ = is_using_device;
  p->is_pinned_ = Bool_false;
  p->is_alias_  = Bool_false;
}

/*---------------------------------------------------------------------------*/

void Pointer_create_alias( Pointer* p,
                           Pointer* source,
                           size_t   base,
                           size_t   n )
{
  Assert( p );
  Assert( source );
  Assert( base+1 >= 1 );
  Assert( n+1 >= 1 );
  Assert( base+n <= source->n_ );

  p->h_ = NULL;
  if( source->h_ )
  {
    p->h_ = source->h_ + base;
  }

  p->d_ = NULL;
  if( source->h_ )
  {
    p->d_ = source->d_ + base;
  }

  p->n_               = n;
  p->is_using_device_ = source->is_using_device_;
  p->is_pinned_       = source->is_pinned_;
  p->is_alias_        = Bool_true;
}

/*---------------------------------------------------------------------------*/

void Pointer_set_pinned( Pointer* p,
                         Bool_t   is_pinned )
{
  Assert( p );
  Assert( ! p->h_
              ? "Currently cannot change pinnedness of allocated array" : 0 );
  Assert( ! p->is_alias_ );

  p->is_pinned_ = is_pinned;
}

/*===========================================================================*/
/*---Pseudo-destructor---*/

void Pointer_destroy( Pointer* p )
{
  Assert( p );

  if( p->h_ && ! p->is_alias_ )
  {
    if( p->is_pinned_ && p->is_using_device_ )
    {
      free_host_pinned_P( p->h_ );
    }
    else
    {
      free_host_P( p->h_ );
    }
  }
  p->h_ = NULL;

  if( p->d_ && ! p->is_alias_ )
  {
    free_device_P( p->d_ );
  }
  p->d_ = NULL;

  p->n_ = 0;
  p->is_using_device_ = Bool_false;
  p->is_pinned_       = Bool_false;
}

/*===========================================================================*/
/*---De/allocate memory---*/

void Pointer_allocate_h_( Pointer* p )
{
  Assert( p );
  Assert( ! p->is_alias_ );
  Assert( ! p->h_ );

  if( p->is_pinned_ && p->is_using_device_ )
  {
    p->h_ = malloc_host_pinned_P( p->n_ );
  }
  else
  {
    p->h_ = malloc_host_P( p->n_ );
  }
  Assert( p->h_ );
}

/*---------------------------------------------------------------------------*/

void Pointer_allocate_d_( Pointer* p )
{
  Assert( p );
  Assert( ! p->is_alias_ );
  Assert( ! p->d_ );

  if( p->is_using_device_ )
  {
    p->d_ = malloc_device_P( p->n_ );

    Assert( p->d_ );
  }
}

/*---------------------------------------------------------------------------*/

void Pointer_allocate( Pointer* p )
{
  Assert( p );
  Assert( ! p->is_alias_ );

  Pointer_allocate_h_( p );
  Pointer_allocate_d_( p );
}

/*---------------------------------------------------------------------------*/

void Pointer_deallocate_h_( Pointer* p )
{
  Assert( p );
  Assert( ! p->is_alias_ );
  Assert( p->h_ );

  if( p->is_pinned_ && p->is_using_device_ )
  {
    free_host_pinned_P( p->h_ );
  }
  else
  {
    free_host_P( p->h_ );
  }
  p->h_ = NULL;
}

/*---------------------------------------------------------------------------*/

void Pointer_deallocate_d_( Pointer* p )
{
  Assert( p );
  Assert( ! p->is_alias_ );

  if( p->is_using_device_ )
  {
    Assert( p->d_ );

    free_device_P( p->d_ );

    p->d_ = NULL;
  }
}

/*---------------------------------------------------------------------------*/

void Pointer_deallocate( Pointer* p )
{
  Assert( p );
  Assert( ! p->is_alias_ );

  Pointer_deallocate_h_( p );
  Pointer_deallocate_d_( p );
}

/*===========================================================================*/
/*---Copy memory---*/

void Pointer_update_h( Pointer* p )
{
  Assert( p );

  if( p->is_using_device_ )
  {
    cuda_copy_device_to_host_P( p->h_, p->d_, p->n_ );
  }
}

/*---------------------------------------------------------------------------*/

void Pointer_update_d( Pointer* p )
{
  Assert( p );

  if( p->is_using_device_ )
  {
    cuda_copy_host_to_device_P( p->d_, p->h_, p->n_ );
  }
}

/*---------------------------------------------------------------------------*/

void Pointer_update_h_stream( Pointer* p, Stream_t stream )
{
  Assert( p );

  if( p->is_using_device_ )
  {
    cuda_copy_device_to_host_stream_P( p->h_, p->d_, p->n_, stream );
  }
}

/*---------------------------------------------------------------------------*/

void Pointer_update_d_stream( Pointer* p, Stream_t stream )
{
  Assert( p );

  if( p->is_using_device_ )
  {
    cuda_copy_host_to_device_stream_P( p->d_, p->h_, p->n_, stream );
  }
}

/*===========================================================================*/
  
#ifdef __cplusplus
} /*---extern "C"---*/
#endif

/*---------------------------------------------------------------------------*/
