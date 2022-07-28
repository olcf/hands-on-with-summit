/*---------------------------------------------------------------------------*/
/*!
 * \file   arguments.h
 * \author Wayne Joubert
 * \date   Wed Mar 12 14:19:26 EDT 2014
 * \brief  Declarations for Arguments struct.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

/*=============================================================================

The Arguments struct serves as a mechanism for specifying package-wide
execution options.
It is internally based on the standard argc/argv specification of
command line arguments.
Its entries are "read-once": when an entry is read, it is deleted.

=============================================================================*/

#ifndef _arguments_h_
#define _arguments_h_

#include <stddef.h>

#include "types.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/* Struct to hold command line arguments being processed---*/

typedef struct
{
  int    argc;
  char** argv_unconsumed; /*---Working copy of argument list---*/
  char*  argstring;
} Arguments;

/*===========================================================================*/
/*---Null object---*/

Arguments Arguments_null(void);

/*===========================================================================*/
/* Pseudo-constructor for Arguments struct---*/

void Arguments_create( Arguments* args,
                       int        argc,
                       char**     argv );

/*===========================================================================*/
/* Pseudo-constructor that takes a string instead of an args array---*/

void Arguments_create_from_string( Arguments*  args,
                                   const char* argstring );

/*===========================================================================*/
/* Pseudo-destructor for Arguments struct---*/

void Arguments_destroy( Arguments* args );

/*===========================================================================*/
/* Determine whether an argument with a given name exists---*/

Bool_t Arguments_exists( const Arguments*  args,
                         const char* arg_name );

/*===========================================================================*/
/* Process an argument of type int, remove from list---*/

int Arguments_consume_int_( Arguments*  args,
                            const char* arg_name );

/*===========================================================================*/
/* Consume an argument of type int, if not present then set to a default---*/

int Arguments_consume_int_or_default( Arguments*  args,
                                      const char* arg_name,
                                      int         default_value );

/*===========================================================================*/
/* Determine whether all arguments have been consumed---*/

Bool_t Arguments_are_all_consumed( const Arguments* args );

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_arguments_h_---*/

/*---------------------------------------------------------------------------*/
