/*---------------------------------------------------------------------------*/
/*!
 * \file   tester.c
 * \author Wayne Joubert
 * \date   Wed May 22 11:22:14 EDT 2013
 * \brief  Tester for sweep miniapp.
 * \note   Copyright (C) 2013 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#include <stdio.h>

#include "arguments.h"
#include "env.h"
#include "definitions.h"
#include "dimensions.h"
#include "pointer.h"
#include "quantities.h"
#include "array_operations.h"
#include "sweeper.h"

#include "runner.h"

#define MAX_LINE_LEN 1024

/*===========================================================================*/

static void compare_runs_helper( Env* env, int* ntest,
    int* ntest_passed, const char* string_common, const char* string1, const char* string2 )
{
  char argstring1[MAX_LINE_LEN];
  char argstring2[MAX_LINE_LEN];

  sprintf( argstring1, "%s %s", string_common, string1 );
  sprintf( argstring2, "%s %s", string_common, string2 );

  const Bool_t result = compare_runs( argstring1, argstring2, env );

  *ntest += 1;
  *ntest_passed += result ? 1 : 0;
}

/*===========================================================================*/
/*---Tester: Serial---*/

static void test_serial( Env* env, int* ntest, int* ntest_passed )
{
#ifdef SWEEPER_KBA
#ifndef USE_MPI
#ifndef USE_OPENMP
#ifndef USE_CUDA
  const Bool_t do_tests = Bool_true;
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif

  if( do_tests )
  {
    int key = 0;
    for( key=0; key<2; ++key )
    {
      char string_common[MAX_LINE_LEN];
      sprintf( string_common,
               "--ncell_x %i --ncell_y  %i --ncell_z  %i --ne 3 --na 7",
               2+3*key, 1+3*key, 2+3*key );
      int ncell_x_per_subblock = 0;
      for( ncell_x_per_subblock=1; ncell_x_per_subblock<=3;
                                 ++ncell_x_per_subblock)
      {
      int ncell_y_per_subblock = 0;
      for( ncell_y_per_subblock=1; ncell_y_per_subblock<=3;
                                 ++ncell_y_per_subblock)
      {
      int ncell_z_per_subblock = 0;
      for( ncell_z_per_subblock=1; ncell_z_per_subblock<=3;
                                 ++ncell_z_per_subblock)
      {
        char string1[] = "";
        char string2[MAX_LINE_LEN];
        sprintf( string2, "--ncell_x_per_subblock %i "
          "--ncell_y_per_subblock %i --ncell_z_per_subblock %i",
          ncell_x_per_subblock, ncell_y_per_subblock, ncell_z_per_subblock );
        compare_runs_helper( env, ntest, ntest_passed, string_common,
          string1, string2 );
      }
      }
      }
    }
  }
}

/*===========================================================================*/
/*---Tester: OpenMP---*/

static void test_openmp( Env* env, int* ntest, int* ntest_passed )
{
#ifdef SWEEPER_KBA
#ifndef USE_MPI
#ifdef USE_OPENMP_THREADS
#ifndef USE_CUDA
  const Bool_t do_tests = Bool_true;
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif

  if( do_tests )
  {
    char string_pat_1[] = "--ncell_x 1 --ncell_y 2 --ncell_z  1 --ne 1 --na 1 "
      "--ncell_x_per_subblock 1 --ncell_y_per_subblock 1 "
      "--ncell_z_per_subblock 1 --nthread_y %i";

    char string1_1[MAX_LINE_LEN];
    sprintf( string1_1, string_pat_1, 1 );

    char string2_1[MAX_LINE_LEN];
    sprintf( string2_1, string_pat_1, 2 );

    compare_runs_helper( env, ntest, ntest_passed, "", string1_1, string2_1 );

    /*-----*/

    char string_pat_2[] = "--ncell_x 5 --ncell_y 4 --ncell_z 5 --ne 17 --na 10 "
      "--nthread_e %i --nthread_octant %i";

    char string1_2[MAX_LINE_LEN];
    sprintf( string1_2, string_pat_2, 1, 1 );

    int nthread_e = 0;
    int nthread_octant = 0;
    int nthread_octant_key = 0;

    for( nthread_e=1; nthread_e<=5; ++nthread_e )
    {
    for( nthread_octant_key=0; nthread_octant_key<=3; ++nthread_octant_key )
    {
      nthread_octant = 1 << nthread_octant_key;
      char string2_2[MAX_LINE_LEN];
      sprintf( string2_2, string_pat_2, nthread_e, nthread_octant );

      compare_runs_helper( env, ntest, ntest_passed, "", string1_2, string2_2 );
    }
    }

    /*-----*/

    const int ncell_x = 3;
    const int ncell_y = 4;
    const int ncell_z = 2;

    char string_common[MAX_LINE_LEN];
    sprintf( string_common, "--ncell_x %i --ncell_y %i --ncell_z %i "
      " --ne 2 --na 1",  ncell_x, ncell_y, ncell_z );

    int ncell_x_per_subblock = 0;
    int ncell_y_per_subblock = 0;
    int ncell_z_per_subblock = 0;
    int nthread_y = 0;
    int nthread_z = 0;

    for( ncell_x_per_subblock=1; ncell_x_per_subblock<=ncell_x+1;
                               ++ncell_x_per_subblock )
    {
    for( ncell_y_per_subblock=1; ncell_y_per_subblock<=ncell_y+1;
                               ++ncell_y_per_subblock )
    {
    for( ncell_z_per_subblock=1; ncell_z_per_subblock<=ncell_z+1;
                               ++ncell_z_per_subblock )
    {
    for( nthread_y=1; nthread_y<=2; ++nthread_y )
    {
    for( nthread_z=1; nthread_z<=2; ++nthread_z )
    {
    for( nthread_e=1; nthread_e<=2; ++nthread_e )
    {
    for( nthread_octant=1; nthread_octant<=2; ++nthread_octant )
    {
      char string1[] = "";
      char string2[MAX_LINE_LEN];
      sprintf( string2, "--ncell_x_per_subblock %i --ncell_y_per_subblock %i "
        "--ncell_z_per_subblock %i "
        "--nthread_y %i --nthread_z %i --nthread_e %i --nthread_octant %i",
        ncell_x_per_subblock, ncell_y_per_subblock, ncell_z_per_subblock,
        nthread_y, nthread_z, nthread_e, nthread_octant );
      compare_runs_helper( env, ntest, ntest_passed, string_common,
        string1, string2 );
    }
    }
    }
    }
    }
    }
    }
  }
}

/*===========================================================================*/
/*---Tester: OpenMP tasks`---*/

static void test_openmp_tasks( Env* env, int* ntest, int* ntest_passed )
{
#ifdef SWEEPER_KBA
#ifndef USE_MPI
#ifdef USE_OPENMP_TASKS
#ifndef USE_CUDA
  const Bool_t do_tests = Bool_true;
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif

  if( do_tests )
  {
    char string1[] = "--ncell_x 9 --ncell_y 13 --ncell_z 3 "
                     "--ne 5 --na 4 --nblock_z 1";

    int nthread_e = 0;
    int nthread_octant_key = 0;
    int ncell_x_per_subblock_key = 0;
    int ncell_y_per_subblock_key = 0;
    int ncell_z_per_subblock_key = 0;

    for( nthread_e=1; nthread_e<=2; ++nthread_e )
    {

    for( nthread_octant_key=0; nthread_octant_key<=3; ++nthread_octant_key )
    {
      const int nthread_octant = 1 << nthread_octant_key;

    for(   ncell_x_per_subblock_key=0; ncell_x_per_subblock_key<2;
         ++ncell_x_per_subblock_key )
    {
      const int ncell_x_per_subblock = ncell_x_per_subblock_key==0 ? 1 : 3;

    for(   ncell_y_per_subblock_key=0; ncell_y_per_subblock_key<2;
         ++ncell_y_per_subblock_key )
    {
      const int ncell_y_per_subblock = ncell_y_per_subblock_key==0 ? 1 : 5;

    for(   ncell_z_per_subblock_key=0; ncell_z_per_subblock_key<2;
         ++ncell_z_per_subblock_key )
    {
      const int ncell_z_per_subblock = ncell_z_per_subblock_key==0 ? 1 : 2;

      char string2[MAX_LINE_LEN];
      sprintf( string2, "%s --nthread_e %i "
                           "--nthread_octant %i "
                           "--ncell_x_per_subblock %i "
                           "--ncell_y_per_subblock %i "
                           "--ncell_z_per_subblock %i ",
               string1, nthread_e, nthread_octant, ncell_x_per_subblock,
               ncell_y_per_subblock, ncell_z_per_subblock );
      compare_runs_helper( env, ntest, ntest_passed, "", string1, string2 );
    }
    }
    }
    }
    }
  }
}

/*===========================================================================*/
/*---Tester: CUDA---*/

static void test_cuda( Env* env, int* ntest, int* ntest_passed )
{
#ifdef SWEEPER_KBA
#ifdef USE_MPI
#ifndef USE_OPENMP
#ifdef USE_CUDA
  const Bool_t do_tests = Bool_true;
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif

  if( do_tests )
  {
    char string_common_1[] = "--ncell_x 2 --ncell_y 3 --ncell_z 4 "
                             "--ne 20 --na 5 --nblock_z 2";

    int nthread_octant = 0;
    int nthread_octant_key = 0;
    for( nthread_octant_key=0; nthread_octant_key<=3; ++nthread_octant_key )
    {
      nthread_octant = 1 << nthread_octant_key;
      char string2[MAX_LINE_LEN];
      sprintf( string2, "--is_using_device 1 "
        "--nthread_e %i --nthread_octant %i", 1, nthread_octant );
      compare_runs_helper( env, ntest, ntest_passed, string_common_1,
        "", string2 );
    }

    /*-----*/

    int nthread_e = 0;
    for( nthread_e=1; nthread_e<=20; ++nthread_e )
    {
      char string2[MAX_LINE_LEN];
      sprintf( string2, "--is_using_device 1 "
        "--nthread_e %i --nthread_octant %i", nthread_e, 8 );
      compare_runs_helper( env, ntest, ntest_passed, string_common_1,
        "", string2 );
    }

    /*-----*/

    const int ncell_x = 3;
    const int ncell_y = 4;
    const int ncell_z = 2;

    char string_common_2[MAX_LINE_LEN];
    sprintf( string_common_2, "--ncell_x %i --ncell_y %i --ncell_z %i "
      " --ne 2 --na 32",  ncell_x, ncell_y, ncell_z );

    int ncell_x_per_subblock = 0;
    int ncell_y_per_subblock = 0;
    int ncell_z_per_subblock = 0;
    int nthread_y = 0;
    int nthread_z = 0;

    for( ncell_x_per_subblock=1; ncell_x_per_subblock<=ncell_x+1;
                               ++ncell_x_per_subblock )
    {
    for( ncell_y_per_subblock=1; ncell_y_per_subblock<=ncell_y+1;
                               ++ncell_y_per_subblock )
    {
    for( ncell_z_per_subblock=1; ncell_z_per_subblock<=ncell_z+1;
                               ++ncell_z_per_subblock )
    {
    for( nthread_y=1; nthread_y<=2; ++nthread_y )
    {
    for( nthread_z=1; nthread_z<=2; ++nthread_z )
    {
    for( nthread_e=1; nthread_e<=2; ++nthread_e )
    {
    for( nthread_octant=1; nthread_octant<=2; ++nthread_octant )
    {
      char string1[] = "";
      char string2[MAX_LINE_LEN];
      sprintf( string2, "--ncell_x_per_subblock %i --ncell_y_per_subblock %i "
        "--ncell_z_per_subblock %i "
        "--nthread_y %i --nthread_z %i --nthread_e %i --nthread_octant %i "
        "--is_using_device 1",
        ncell_x_per_subblock, ncell_y_per_subblock, ncell_z_per_subblock,
        nthread_y, nthread_z, nthread_e, nthread_octant );
      compare_runs_helper( env, ntest, ntest_passed, string_common_2,
        string1, string2 );
    }
    }
    }
    }
    }
    }
    }
  }
}

/*===========================================================================*/
/*---Tester: MPI---*/

static void test_mpi( Env* env, int* ntest, int* ntest_passed )
{
#ifdef SWEEPER_KBA
#ifdef USE_MPI
#ifndef USE_OPENMP
#ifndef USE_CUDA
  const Bool_t do_tests = Bool_true;
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif

  if( do_tests )
  {
    const char* string_common_1 = "--ncell_x  5 --ncell_y  4 --ncell_z  5"
                                  " --ne 7 --na 10";

    compare_runs_helper( env, ntest, ntest_passed, string_common_1,
        "--nproc_x 1 --nproc_y 1 --nblock_z 1",
        "--nproc_x 2 --nproc_y 1 --nblock_z 1" );

    compare_runs_helper( env, ntest, ntest_passed, string_common_1,
        "--nproc_x 1 --nproc_y 1 --nblock_z 1",
        "--nproc_x 1 --nproc_y 2 --nblock_z 1" );

    const char* string_common_2 = "--ncell_x  5 --ncell_y  4 --ncell_z  6"
                                  " --ne 7 --na 10";

    compare_runs_helper( env, ntest, ntest_passed, string_common_2,
        "--nproc_x 1 --nproc_y 1 --nblock_z 1",
        "--nproc_x 4 --nproc_y 4 --nblock_z 2" );

    const char* string_common_3 = "--ncell_x  5 --ncell_y  4 --ncell_z  6"
                                  " --ne 7 --na 10 --is_face_comm_async 0";

    compare_runs_helper( env, ntest, ntest_passed, string_common_3,
        "--nproc_x 1 --nproc_y 1 --nblock_z 1",
        "--nproc_x 4 --nproc_y 4 --nblock_z 2" );

    const char* string_common_4 = "--ncell_x 5 --ncell_y 8 --ncell_z 16"
                                  " --ne 9 --na 12";

    compare_runs_helper( env, ntest, ntest_passed, string_common_4,
        "--nproc_x 4 --nproc_y 4 --nblock_z 1",
        "--nproc_x 4 --nproc_y 4 --nblock_z 2" );

    compare_runs_helper( env, ntest, ntest_passed, string_common_4,
        "--nproc_x 4 --nproc_y 4 --nblock_z 2",
        "--nproc_x 4 --nproc_y 4 --nblock_z 4" );
  }
}

/*===========================================================================*/
/*---Tester: MPI + CUDA---*/

static void test_mpi_cuda( Env* env, int* ntest, int* ntest_passed )
{
#ifdef SWEEPER_KBA
#ifdef USE_MPI
#ifndef USE_OPENMP
#ifdef USE_CUDA
  const Bool_t do_tests = Bool_true;
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif

  if( do_tests )
  {
    char string_common[] = "--ncell_x 3 --ncell_y 5 --ncell_z 6 "
      "--ne 2 --na 5 --nblock_z 2";

    int nproc_x = 0;
    int nproc_y = 0;
    int nthread_octant = 0;
    int nthread_octant_key = 0;

    for( nproc_x=1; nproc_x<=2; ++nproc_x )
    {
    for( nproc_y=1; nproc_y<=2; ++nproc_y )
    {
    for( nthread_octant_key=0; nthread_octant_key<=3; ++nthread_octant_key )
    {
      nthread_octant = 1 << nthread_octant_key;
      char string1[] = "";
      char string2[MAX_LINE_LEN];
      sprintf( string2, "--is_using_device 1 --nproc_x %i --nproc_y %i "
        "--nthread_e 3 --nthread_octant %i",
        nproc_x, nproc_y, nthread_octant );
      compare_runs_helper( env, ntest, ntest_passed, string_common,
        string1, string2 );
    }
    }
    }
  }
}

/*===========================================================================*/
/*---Tester: Variants---*/

static void test_variants( Env* env, int* ntest, int* ntest_passed )
{
#ifndef SWEEPER_KBA
#ifndef USE_MPI
#ifndef USE_OPENMP
#ifndef USE_CUDA
  const Bool_t do_tests = Bool_true;
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif
#else
  const Bool_t do_tests = Bool_false;
#endif

  if( do_tests )
  {
    char string_common[] = "--ncell_x 4 --ncell_y 3 --ncell_z 5 --ne 11 --na 7";

    int niterations = 0;

    for( niterations=1; niterations<=2; ++niterations )
    {
      char string1[] = "";
      char string2[MAX_LINE_LEN];
      sprintf( string2, "--niterations %i", niterations );
      compare_runs_helper( env, ntest, ntest_passed, string_common,
        string1, string2 );
    }
  }
}

/*===========================================================================*/
/*---Tester---*/

static void tester( Env* env )
{
  int ntest = 0;
  int ntest_passed = 0;

  test_serial( env, &ntest, &ntest_passed );

  test_openmp( env, &ntest, &ntest_passed );

  test_openmp_tasks( env, &ntest, &ntest_passed );

  test_mpi( env, &ntest, &ntest_passed );

  test_cuda( env, &ntest, &ntest_passed );

  test_mpi_cuda( env, &ntest, &ntest_passed );

  test_variants( env, &ntest, &ntest_passed );

  if( Env_is_proc_master( env ) )
  {
    printf( "TESTS %i    PASSED %i    FAILED %i\n",
            ntest, ntest_passed, ntest-ntest_passed );
  }
}

/*===========================================================================*/
/*---Main---*/

int main( int argc, char** argv )
{
  /*---Declarations---*/
  Env env = Env_null();

  /*---Initialize for execution---*/

  /*---NOTE: env is only partially initialized by this---*/

  Env_initialize( &env, argc, argv );

  /*---Do testing---*/

  tester( &env );

  /*---Finalize execution---*/

  Env_finalize( &env );

} /*---main---*/

/*---------------------------------------------------------------------------*/
