/*---------------------------------------------------------------------------*/
/*!
 * \file   faces_kba.h
 * \author Wayne Joubert
 * \date   Mon May 12 11:53:27 EDT 2014
 * \brief  Declarations for managing sweeper faces.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _faces_kba_h_
#define _faces_kba_h_

#include "env.h"
#include "pointer.h"
#include "definitions.h"
#include "dimensions.h"
#include "quantities.h"
#include "stepscheduler_kba.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Struct with face info---*/

typedef struct
{
  Pointer          facexy0;

  Pointer          facexz0;
  Pointer          facexz1;
  Pointer          facexz2;

  Pointer          faceyz0;
  Pointer          faceyz1;
  Pointer          faceyz2;

  Request_t        request_send_xz[NOCTANT];
  Request_t        request_send_yz[NOCTANT];
  Request_t        request_recv_xz[NOCTANT];
  Request_t        request_recv_yz[NOCTANT];

  int              noctant_per_block;

  Bool_t           is_face_comm_async;
} Faces;

/*===========================================================================*/
/*---Pseudo-constructor for Faces struct---*/

void Faces_create( Faces*      faces,
                   Dimensions  dims_b,
                   int         noctant_per_block,
                   Bool_t      is_face_comm_async,
                   Env*        env );

/*===========================================================================*/
/*---Pseudo-destructor for Faces struct---*/

void Faces_destroy( Faces* faces );

/*===========================================================================*/
/*---Is face communication done asynchronously---*/

static int Faces_is_face_comm_async( Faces* faces )
{
  return faces->is_face_comm_async;
}

/*===========================================================================*/
/*---Selectors for faces---*/

/*---The xz and yz face arrays form a circular buffer of length three.
     Three are needed because at any step there may be a send, a receive, and a
     block-sweep-compute in-flight.
---*/

static Pointer* Faces_facexy( Faces* faces, int i )
{
  Assert( faces != NULL );
  Assert( i >= 0 && i < 1 );
  return & faces->facexy0;
}

/*---------------------------------------------------------------------------*/

static Pointer* Faces_facexz( Faces* faces, int i )
{
  Assert( faces != NULL );
  Assert( i >= 0 && i < ( Faces_is_face_comm_async( faces ) ? NDIM : 1 ) );
  Pointer* facesxz[NDIM] = { & faces->facexz0,
                             & faces->facexz1,
                             & faces->facexz2 };
  return facesxz[i];
}

/*---------------------------------------------------------------------------*/

static Pointer* Faces_faceyz( Faces* faces, int i )
{
  Assert( faces != NULL );
  Assert( i >= 0 && i < ( Faces_is_face_comm_async( faces ) ? NDIM : 1 ) );
  Pointer* facesyz[NDIM] = { & faces->faceyz0,
                             & faces->faceyz1,
                             & faces->faceyz2 };
  return facesyz[i];
}

/*---------------------------------------------------------------------------*/


static Pointer* Faces_facexy_step( Faces* faces, int step )
{
  Assert( faces != NULL );
  Assert( step >= -1 );
  return Faces_facexy( faces, 0 );
}

/*---------------------------------------------------------------------------*/

static Pointer* Faces_facexz_step( Faces* faces, int step )
{
  Assert( faces != NULL );
  Assert( step >= -1 );

  return Faces_facexz( faces,
                       Faces_is_face_comm_async( faces ) ? (step+3)%3 : 0 );
}

/*---------------------------------------------------------------------------*/

static Pointer* Faces_faceyz_step( Faces* faces, int step )
{
  Assert( faces != NULL );
  Assert( step >= -1 );

  return Faces_faceyz( faces,
                       Faces_is_face_comm_async( faces ) ? (step+3)%3 : 0 );
}

/*===========================================================================*/
/*---Communicate faces computed at step, used at step+1---*/

void Faces_communicate_faces(
  Faces*          faces,
  StepScheduler*  stepscheduler,
  Dimensions      dims_b,
  int             step,
  Env*            env );

/*===========================================================================*/
/*---Asynchronously send faces computed at step, used at step+1: start---*/

void Faces_send_faces_start(
  Faces*          faces,
  StepScheduler*  stepscheduler,
  Dimensions      dims_b,
  int             step,
  Env*            env );

/*===========================================================================*/
/*---Asynchronously send faces computed at step, used at step+1: end---*/

void Faces_send_faces_end(
  Faces*          faces,
  StepScheduler*  stepscheduler,
  Dimensions      dims_b,
  int             step,
  Env*            env );

/*===========================================================================*/
/*---Asynchronously recv faces computed at step, used at step+1: start---*/

void Faces_recv_faces_start(
  Faces*          faces,
  StepScheduler*  stepscheduler,
  Dimensions      dims_b,
  int             step,
  Env*            env );

/*===========================================================================*/
/*---Asynchronously recv faces computed at step, used at step+1: end---*/

void Faces_recv_faces_end(
  Faces*          faces,
  StepScheduler*  stepscheduler,
  Dimensions      dims_b,
  int             step,
  Env*            env );

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_faces_kba_h_---*/

/*---------------------------------------------------------------------------*/

