/*---------------------------------------------------------------------------*/
/*!
 * \file   faces_kba.c
 * \author Wayne Joubert
 * \date   Mon May 12 11:53:27 EDT 2014
 * \brief  Definitions for managing sweeper faces.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#include "env.h"
#include "faces_kba.h"
#include "array_operations.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Pseudo-constructor for Faces struct---*/

void Faces_create( Faces*      faces,
                   Dimensions  dims_b,
                   int         noctant_per_block,
                   Bool_t      is_face_comm_async,
                   Env*        env )
{
  int i = 0;

  faces->noctant_per_block  = noctant_per_block;
  faces->is_face_comm_async = is_face_comm_async;

  /*====================*/
  /*---Allocate faces---*/
  /*====================*/

  Pointer_create(       Faces_facexy( faces, 0 ),
    Dimensions_size_facexy( dims_b, NU, noctant_per_block ),
    Env_cuda_is_using_device( env ) );
  Pointer_set_pinned( Faces_facexy( faces, 0 ), Bool_true );
  Pointer_allocate(     Faces_facexy( faces, 0 ) );

  for( i = 0; i < ( Faces_is_face_comm_async( faces ) ? NDIM : 1 ); ++i )
  {
    Pointer_create(       Faces_facexz( faces, i ),
      Dimensions_size_facexz( dims_b, NU, noctant_per_block ),
      Env_cuda_is_using_device( env ) );
    Pointer_set_pinned( Faces_facexz( faces, i ), Bool_true );

    Pointer_create(       Faces_faceyz( faces, i ),
      Dimensions_size_faceyz( dims_b, NU, noctant_per_block ),
      Env_cuda_is_using_device( env ) );
    Pointer_set_pinned( Faces_faceyz( faces, i ), Bool_true );
  }

  for( i = 0; i < ( Faces_is_face_comm_async( faces ) ? NDIM : 1 ); ++i )
  {
    Pointer_allocate( Faces_facexz( faces, i ) );
    Pointer_allocate( Faces_faceyz( faces, i ) );
  }
}

/*===========================================================================*/
/*---Pseudo-destructor for Faces struct---*/

void Faces_destroy( Faces* faces )
{
  int i = 0;

  /*====================*/
  /*---Deallocate faces---*/
  /*====================*/

  Pointer_destroy( Faces_facexy( faces, 0 ) );

  for( i = 0; i < ( Faces_is_face_comm_async( faces ) ? NDIM : 1 ); ++i )
  {
    Pointer_destroy( Faces_facexz( faces, i ) );
    Pointer_destroy( Faces_faceyz( faces, i ) );
  }
}
/*===========================================================================*/
/*---Communicate faces computed at step, used at step+1---*/

void Faces_communicate_faces(
  Faces*          faces,
  StepScheduler*  stepscheduler,
  Dimensions      dims_b,
  int             step,
  Env*            env )
{
  Assert( ! Faces_is_face_comm_async( faces ) );

  const int proc_x = Env_proc_x_this( env );
  const int proc_y = Env_proc_y_this( env );

  const size_t size_facexz_per_octant = Dimensions_size_facexz( dims_b,
                 NU, faces->noctant_per_block ) / faces->noctant_per_block;
  const size_t size_faceyz_per_octant = Dimensions_size_faceyz( dims_b,
                 NU, faces->noctant_per_block ) / faces->noctant_per_block;

  /*---Allocate temporary face buffers---*/

  P* __restrict__ buf_xz  = malloc_host_P( size_facexz_per_octant );
  P* __restrict__ buf_yz  = malloc_host_P( size_faceyz_per_octant );

  /*---Loop over octants---*/

  int octant_in_block = 0;

  for( octant_in_block=0; octant_in_block<faces->noctant_per_block;
                                                            ++octant_in_block )
  {
    /*---Communicate +/-X, +/-Y---*/

    int axis = 0;

    for( axis=0; axis<2; ++axis )  /*---Loop: X, Y---*/
    {
      const Bool_t axis_x = axis==0;
      const Bool_t axis_y = axis==1;

      const int proc_axis = axis_x ? proc_x : proc_y;

      const size_t    size_face_per_octant    = axis_x ? size_faceyz_per_octant
                                                       : size_facexz_per_octant;
      P* __restrict__ buf                     = axis_x ? buf_yz
                                                       : buf_xz;
      P* __restrict__ face_per_octant = axis_x ?
        ref_faceyz( Pointer_h( Faces_faceyz_step( faces, step ) ),
                    dims_b, NU, faces->noctant_per_block,
                    0, 0, 0, 0, 0, octant_in_block ) :
        ref_facexz( Pointer_h( Faces_facexz_step( faces, step ) ),
                    dims_b, NU, faces->noctant_per_block,
                     0, 0, 0, 0, 0, octant_in_block );

      int dir_ind = 0;

      for( dir_ind=0; dir_ind<2; ++dir_ind ) /*---Loop: up, down---*/
      {
        const int dir = dir_ind==0 ? DIR_UP*1 : DIR_DN*1;
        const int inc_x = axis_x ? Dir_inc( dir ) : 0;
        const int inc_y = axis_y ? Dir_inc( dir ) : 0;

        /*---Determine whether to communicate---*/

        Bool_t const do_send = StepScheduler_must_do_send(
                   stepscheduler, step, axis, dir_ind, octant_in_block, env );

        Bool_t const do_recv = StepScheduler_must_do_recv(
                   stepscheduler, step, axis, dir_ind, octant_in_block, env );

        /*---Communicate as needed - red/black coloring to avoid deadlock---*/

        int color = 0;

        Bool_t use_buf = Bool_false;

        for( color=0; color<2; ++color )
        {
          if( color == 0 )
          {
            if( proc_axis % 2 == 0 )
            {
              if( do_send )
              {
                const int proc_other
                                 = Env_proc( env, proc_x+inc_x, proc_y+inc_y );
                Env_send_P( env, face_per_octant, size_face_per_octant,
                            proc_other, Env_tag( env )+octant_in_block );
              }
            }
            else
            {
              if( do_recv )
              {
                const int proc_other
                                 = Env_proc( env, proc_x-inc_x, proc_y-inc_y );
                /*---save copy else color 0 recv will destroy color 1 send---*/
                copy_vector( buf, face_per_octant, size_face_per_octant );
                use_buf = Bool_true;
                Env_recv_P( env, face_per_octant, size_face_per_octant,
                            proc_other, Env_tag( env )+octant_in_block );
              }
            }
          }
          else /*---if color---*/
          {
            if( proc_axis % 2 == 0 )
            {
              if( do_recv )
              {
                const int proc_other
                                 = Env_proc( env, proc_x-inc_x, proc_y-inc_y );
                Env_recv_P( env, face_per_octant, size_face_per_octant,
                            proc_other, Env_tag( env )+octant_in_block );
              }
            }
            else
            {
              if( do_send )
              {
                const int proc_other
                                 = Env_proc( env, proc_x+inc_x, proc_y+inc_y );
                Env_send_P( env, use_buf ? buf : face_per_octant,
                  size_face_per_octant, proc_other,
                  Env_tag( env )+octant_in_block );
              }
            }
          } /*---if color---*/
        } /*---color---*/
      } /*---dir_ind---*/
    } /*---axis---*/
  } /*---octant_in_block---*/

  /*---Deallocations---*/

  free_host_P( buf_xz );
  free_host_P( buf_yz );
}

/*===========================================================================*/
/*---Asynchronously send faces computed at step, used at step+1: start---*/

void Faces_send_faces_start(
  Faces*          faces,
  StepScheduler*  stepscheduler,
  Dimensions      dims_b,
  int             step,
  Env*            env )
{
  Assert( Faces_is_face_comm_async( faces ) );

  const int proc_x = Env_proc_x_this( env );
  const int proc_y = Env_proc_y_this( env );

  const size_t size_facexz_per_octant = Dimensions_size_facexz( dims_b,
                 NU, faces->noctant_per_block ) / faces->noctant_per_block;
  const size_t size_faceyz_per_octant = Dimensions_size_faceyz( dims_b,
                 NU, faces->noctant_per_block ) / faces->noctant_per_block;

  /*---Loop over octants---*/

  int octant_in_block = 0;

  for( octant_in_block=0; octant_in_block<faces->noctant_per_block;
                                                            ++octant_in_block )
  {
    /*---Communicate +/-X, +/-Y---*/

    int axis = 0;

    for( axis=0; axis<2; ++axis )
    {
      const Bool_t axis_x = axis==0;
      const Bool_t axis_y = axis==1;

      /*---Send values computed on this step---*/

      const size_t    size_face_per_octant    = axis_x ? size_faceyz_per_octant
                                                       : size_facexz_per_octant;
      P* __restrict__ face_per_octant = axis_x ?
        ref_faceyz( Pointer_h( Faces_faceyz_step( faces, step ) ),
                    dims_b, NU, faces->noctant_per_block,
                    0, 0, 0, 0, 0, octant_in_block ) :
        ref_facexz( Pointer_h( Faces_facexz_step( faces, step ) ),
                    dims_b, NU, faces->noctant_per_block,
                    0, 0, 0, 0, 0, octant_in_block );

      int dir_ind = 0;

      for( dir_ind=0; dir_ind<2; ++dir_ind )
      {
        const int dir = dir_ind==0 ? DIR_UP*1 : DIR_DN*1;
        const int inc_x = axis_x ? Dir_inc( dir ) : 0;
        const int inc_y = axis_y ? Dir_inc( dir ) : 0;

        /*---Determine whether to communicate---*/

        Bool_t const do_send = StepScheduler_must_do_send(
                   stepscheduler, step, axis, dir_ind, octant_in_block, env );

        if( do_send )
        {
          const int proc_other = Env_proc( env, proc_x+inc_x, proc_y+inc_y );
          Request_t* request = axis_x ?
                                   & faces->request_send_xz[octant_in_block]
                                 : & faces->request_send_yz[octant_in_block];
          Env_asend_P( env, face_per_octant, size_face_per_octant,
                    proc_other, Env_tag( env )+octant_in_block, request );
        }
      } /*---dir_ind---*/
    } /*---axis---*/
  } /*---octant_in_block---*/
}

/*===========================================================================*/
/*---Asynchronously send faces computed at step, used at step+1: end---*/

void Faces_send_faces_end(
  Faces*          faces,
  StepScheduler*  stepscheduler,
  Dimensions      dims_b,
  int             step,
  Env*            env )
{
  Assert( Faces_is_face_comm_async( faces ) );

  const int proc_x = Env_proc_x_this( env );
  const int proc_y = Env_proc_y_this( env );

  /*---Loop over octants---*/

  int octant_in_block = 0;

  for( octant_in_block=0; octant_in_block<faces->noctant_per_block;
                                                            ++octant_in_block )
  {
    /*---Communicate +/-X, +/-Y---*/

    int axis = 0;

    for( axis=0; axis<2; ++axis )
    {
      const Bool_t axis_x = axis==0;

      int dir_ind = 0;

      for( dir_ind=0; dir_ind<2; ++dir_ind )
      {

        /*---Determine whether to communicate---*/

        Bool_t const do_send = StepScheduler_must_do_send(
                   stepscheduler, step, axis, dir_ind, octant_in_block, env );

        if( do_send )
        {
          Request_t* request = axis_x ?
                                   & faces->request_send_xz[octant_in_block]
                                 : & faces->request_send_yz[octant_in_block];
          Env_wait( env, request );
        }
      } /*---dir_ind---*/
    } /*---axis---*/
  } /*---octant_in_block---*/
}

/*===========================================================================*/
/*---Asynchronously recv faces computed at step, used at step+1: start---*/

void Faces_recv_faces_start(
  Faces*          faces,
  StepScheduler*  stepscheduler,
  Dimensions      dims_b,
  int             step,
  Env*            env )
{
  Assert( Faces_is_face_comm_async( faces ) );

  const int proc_x = Env_proc_x_this( env );
  const int proc_y = Env_proc_y_this( env );

  const size_t size_facexz_per_octant = Dimensions_size_facexz( dims_b,
                 NU, faces->noctant_per_block ) / faces->noctant_per_block;
  const size_t size_faceyz_per_octant = Dimensions_size_faceyz( dims_b,
                 NU, faces->noctant_per_block ) / faces->noctant_per_block;

  /*---Loop over octants---*/

  int octant_in_block = 0;

  for( octant_in_block=0; octant_in_block<faces->noctant_per_block;
                                                            ++octant_in_block )
  {
    /*---Communicate +/-X, +/-Y---*/

    int axis = 0;

    for( axis=0; axis<2; ++axis )
    {
      const Bool_t axis_x = axis==0;
      const Bool_t axis_y = axis==1;

      /*---Receive values computed on the next step---*/

      const size_t    size_face_per_octant    = axis_x ? size_faceyz_per_octant
                                                       : size_facexz_per_octant;
      P* __restrict__ face_per_octant = axis_x ?
        ref_faceyz( Pointer_h( Faces_faceyz_step( faces, step+1 ) ),
                    dims_b, NU, faces->noctant_per_block,
                    0, 0, 0, 0, 0, octant_in_block ) :
        ref_facexz( Pointer_h( Faces_facexz_step( faces, step+1 ) ),
                    dims_b, NU, faces->noctant_per_block,
                    0, 0, 0, 0, 0, octant_in_block );

      int dir_ind = 0;

      for( dir_ind=0; dir_ind<2; ++dir_ind )
      {
        const int dir = dir_ind==0 ? DIR_UP*1 : DIR_DN*1;
        const int inc_x = axis_x ? Dir_inc( dir ) : 0;
        const int inc_y = axis_y ? Dir_inc( dir ) : 0;

        /*---Determine whether to communicate---*/

        Bool_t const do_recv = StepScheduler_must_do_recv(
                   stepscheduler, step, axis, dir_ind, octant_in_block, env );

        if( do_recv )
        {
          const int proc_other = Env_proc( env, proc_x-inc_x, proc_y-inc_y );
          Request_t* request = axis_x ?
                                   & faces->request_recv_xz[octant_in_block]
                                 : & faces->request_recv_yz[octant_in_block];
          Env_arecv_P( env, face_per_octant, size_face_per_octant,
                    proc_other, Env_tag( env )+octant_in_block, request );
        }
      } /*---dir_ind---*/
    } /*---axis---*/
  } /*---octant_in_block---*/
}

/*===========================================================================*/
/*---Asynchronously recv faces computed at step, used at step+1: end---*/

void Faces_recv_faces_end(
  Faces*          faces,
  StepScheduler*  stepscheduler,
  Dimensions      dims_b,
  int             step,
  Env*            env )
{
  Assert( Faces_is_face_comm_async( faces ) );

  const int proc_x = Env_proc_x_this( env );
  const int proc_y = Env_proc_y_this( env );

  const size_t size_facexz_per_octant = Dimensions_size_facexz( dims_b,
                 NU, faces->noctant_per_block ) / faces->noctant_per_block;
  const size_t size_faceyz_per_octant = Dimensions_size_faceyz( dims_b,
                 NU, faces->noctant_per_block ) / faces->noctant_per_block;

  /*---Loop over octants---*/

  int octant_in_block = 0;

  for( octant_in_block=0; octant_in_block<faces->noctant_per_block;
                                                            ++octant_in_block )
  {
    /*---Communicate +/-X, +/-Y---*/

    int axis = 0;

    for( axis=0; axis<2; ++axis )
    {
      const Bool_t axis_x = axis==0;

      int dir_ind = 0;

      for( dir_ind=0; dir_ind<2; ++dir_ind )
      {
        /*---Determine whether to communicate---*/

        Bool_t const do_recv = StepScheduler_must_do_recv(
                   stepscheduler, step, axis, dir_ind, octant_in_block, env );

        if( do_recv )
        {
          Request_t* request = axis_x ?
                                   & faces->request_recv_xz[octant_in_block]
                                 : & faces->request_recv_yz[octant_in_block];
          Env_wait( env, request );
        }
      } /*---dir_ind---*/
    } /*---axis---*/
  } /*---octant_in_block---*/
}

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

/*---------------------------------------------------------------------------*/
