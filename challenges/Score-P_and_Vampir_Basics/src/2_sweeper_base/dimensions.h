/*---------------------------------------------------------------------------*/
/*!
 * \file   dimensions.h
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Problem dimensions.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _dimensions_h_
#define _dimensions_h_

#include <stddef.h>

#include "env.h"
#include "types.h"
#include "dimensions_kernels.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Null object---*/

Dimensions Dimensions_null(void);

/*===========================================================================*/
/*---Size of state vector---*/

size_t Dimensions_size_state( const Dimensions dims, int nu );

/*===========================================================================*/
/*---Size of state vector in angles space---*/

size_t Dimensions_size_state_angles( const Dimensions dims, int nu );

/*===========================================================================*/
/*---Size of face vectors---*/

size_t Dimensions_size_facexy( const Dimensions dims,
                               int nu,
                               int num_face_octants_allocated );

/*---------------------------------------------------------------------------*/

size_t Dimensions_size_facexz( const Dimensions dims,
                               int nu,
                               int num_face_octants_allocated );

/*---------------------------------------------------------------------------*/

size_t Dimensions_size_faceyz( const Dimensions dims,
                               int nu,
                               int num_face_octants_allocated );

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_dimensions_h_---*/

/*---------------------------------------------------------------------------*/
