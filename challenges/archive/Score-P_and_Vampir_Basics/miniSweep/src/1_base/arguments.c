/*---------------------------------------------------------------------------*/
/*!
 * \file   arguments.c
 * \author Wayne Joubert
 * \date   Wed Mar 12 14:19:26 EDT 2014
 * \brief  Definitions for Arguments struct.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "arguments.h"
#include "types.h"
#include "env_assert.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Null object---*/

Arguments Arguments_null(void)
{
  Arguments result;
  memset( (void*)&result, 0, sizeof(Arguments) );
  return result;
}

/*===========================================================================*/
/* Pseudo-constructor for Arguments struct---*/

void Arguments_create( Arguments* args,
                       int        argc,
                       char**     argv )
{
  Assert( args != NULL );
  Assert( argc > 0 );
  Assert( argv != NULL );
  int i = 0;

  args->argc = argc;
  args->argv_unconsumed = (char**) malloc( argc * sizeof( char* ) );
  args->argstring = 0;

  for( i=0; i<argc; ++i )
  {
    Assert( argv[i] != NULL ? "Null command line argument encountered" : 0);
    args->argv_unconsumed[i] = argv[i];
  }
} /*---Arguments_create---*/

/*===========================================================================*/
/* Pseudo-constructor that takes a string instead of an args array---*/

void Arguments_create_from_string( Arguments*  args,
                                   const char* argstring )
{
  Assert( args != NULL );
  Assert( argstring != NULL );

  size_t len = strlen( argstring );

  args->argstring = (char*) malloc( (len+1) * sizeof( char ) );
  memcpy( args->argstring, argstring, len+1 );

  args->argv_unconsumed = (char**) malloc( len * sizeof( char* ) );

  args->argv_unconsumed[0] = & args->argstring[len];
  args->argc = 1;

  /*---Parse string, convert to args---*/
  Bool_t is_delim_prev = Bool_true;
  int i = 0;
  for( i=0; i<len; ++i )
  {
    if( args->argstring[i] == ' ' || args->argstring[i] == '\t' )
    {
      args->argstring[i] = 0;
    }
    Bool_t is_delim = args->argstring[i] == 0;
    if( is_delim_prev && ! is_delim )
    {
      args->argv_unconsumed[args->argc] = & args->argstring[i];
      args->argc++;
    }
    is_delim_prev = is_delim;
  }
} /*---Arguments_create_from_string---*/

/*===========================================================================*/
/* Pseudo-destructor for Arguments struct---*/

void Arguments_destroy( Arguments* args )
{
  Assert( args != NULL );

  free( (void*) args->argv_unconsumed );
  if( args->argstring )
  {
    free( (void*) args->argstring );
  }
} /*---Arguments_destroy---*/

/*===========================================================================*/
/* Determine whether an argument with a given name exists---*/

Bool_t Arguments_exists( const Arguments*  args,
                         const char* arg_name )
{
  Assert( args != NULL );
  Assert( arg_name != NULL );

  Bool_t result = Bool_false;
  int i = 0;

  for( i=0; i<args->argc; ++i )
  {
    if( args->argv_unconsumed[i] == NULL )
    {
      continue;
    }
    result = result || strcmp( args->argv_unconsumed[i], arg_name ) == 0;
  }

  return result;
}

/*===========================================================================*/
/* Process an argument of type int, remove from list---*/

int Arguments_consume_int_( Arguments*  args,
                            const char* arg_name )
{
  Assert( args != NULL );
  Assert( arg_name != NULL );

  int result = 0;
  Bool_t found = Bool_false;
  if( found ) {} /*---Remove unused var warning---*/
  int i = 0;

  for( i=0; i<args->argc; ++i )
  {
    if( args->argv_unconsumed[i] == NULL )
    {
      continue;
    }
    if( strcmp( args->argv_unconsumed[i], arg_name ) == 0 )
    {
      found = Bool_true;
      args->argv_unconsumed[i] = NULL;
      ++i;
      Insist( i<args->argc );
      result = atoi( args->argv_unconsumed[i] );
      args->argv_unconsumed[i] = NULL;
    }
  }

  Insist( found ? "Invalid use of argument." : 0 );
  return result;
}

/*===========================================================================*/
/* Consume an argument of type int, if not present then set to a default---*/

int Arguments_consume_int_or_default( Arguments*  args,
                                      const char* arg_name,
                                      int         default_value )
{
  Assert( args != NULL );
  Assert( arg_name != NULL );

  return Arguments_exists( args, arg_name ) ?
                     Arguments_consume_int_( args, arg_name ) : default_value;
}

/*===========================================================================*/
/* Determine whether all arguments have been consumed---*/

Bool_t Arguments_are_all_consumed( const Arguments* args )
{
  Assert( args != NULL );

  Bool_t result = Bool_true;
  int i = 0;

  for( i=1; i<args->argc; ++i ) /*---Note: skip the zeroth element---*/
  {
    result = result && args->argv_unconsumed[i] == NULL;
  }

  return result;
}

/*===========================================================================*/
  
#ifdef __cplusplus
} /*---extern "C"---*/
#endif

/*===========================================================================*/
