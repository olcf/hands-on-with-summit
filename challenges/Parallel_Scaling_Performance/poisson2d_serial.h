/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

void poisson2d_serial(int iter_max, double tol)
{
    int iter  = 0;
    double error = 1.0;
   
	#pragma acc data copyin(A_ref[0:NY][0:NX], rhs[0:NY][0:NX]) create(Anew_ref[0:NY][0:NX])
	{
 
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

		#pragma acc kernels
        for (int iy = 1; iy < NY-1; iy++)
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                Anew_ref[iy][ix] = -0.25 * (rhs[iy][ix] - ( A_ref[iy][ix+1] + A_ref[iy][ix-1]
                                                       + A_ref[iy-1][ix] + A_ref[iy+1][ix] ));
                error = fmax( error, fabs(Anew_ref[iy][ix]-A_ref[iy][ix]));
            }
        }
        
		#pragma acc kernels
        for (int iy = 1; iy < NY-1; iy++)
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                A_ref[iy][ix] = Anew_ref[iy][ix];
            }
        }
        
        //Periodic boundary conditions
		#pragma acc kernels
        for( int ix = 1; ix < NX-1; ix++ )
        {
                A_ref[0][ix]      = A_ref[(NY-2)][ix];
                A_ref[(NY-1)][ix] = A_ref[1][ix];
        }
		#pragma acc kernels
        for (int iy = 1; iy < NY-1; iy++)
        {
                A_ref[iy][0]      = A_ref[iy][(NX-2)];
                A_ref[iy][(NX-1)] = A_ref[iy][1];
        }
        
        if((iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }
	
	#pragma acc update self(A_ref)
	} /* pragma acc data */

}
