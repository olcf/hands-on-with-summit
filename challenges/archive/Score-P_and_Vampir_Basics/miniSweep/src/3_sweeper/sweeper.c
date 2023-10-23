/*---------------------------------------------------------------------------*/
/*!
 * \file   sweeper.c
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Definitions for performing a sweep.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#include "sweeper.h"

#ifdef SWEEPER_SIMPLE
#include "sweeper_simple_c.h"
#endif

#ifdef SWEEPER_TILEOCTANTS
#include "sweeper_tileoctants_c.h"
#endif

#ifdef SWEEPER_KBA
#include "sweeper_kba_c.h"
#endif

/*---------------------------------------------------------------------------*/
