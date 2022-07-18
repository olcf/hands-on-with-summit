# galaxy_solution.py
# Author: Michael A. Sandoval
# Simulates orbits of a random distribution of particles (an "infalling galaxy") around a "host galaxy".
# Individual particles only interact with the host galaxy and the nucleus of the infalling galaxy.
# The nucleus of the infalling galaxy only interacts with the host galaxy.

import h5py
import numpy as np
import time as tp
from mpi4py import MPI
from scipy.integrate import odeint

comm = MPI.COMM_WORLD      # Use the world communicator
mpi_rank = comm.Get_rank() # The MPI task ID
mpi_size = comm.Get_size() # Total amount of ranks

def calc_vels_accs(coords, time):
    '''
    Given initial velocities and positions for
    infalling particles (x,y,z notation) and the
    infalling nucleus (x_nuc, y_nuc, z_nuc notation),
    calculates the resulting accelerations from the 
    interactions between the particles, nucleus, 
    and the host (non-infalling) galaxy.

    For simplicity, the nucleus is only influenced
    by the host galaxy (the influence the particles
    have on the nucleus is assumed to be negligible).
    All units in SI units.
    '''

    # Constants
    G = 6.673e-11

    # Initial Positions
    x_nuc, y_nuc, z_nuc = coords[0], coords[1], coords[2]
    x, y, z = coords[3], coords[4], coords[5]

    # Initial Velocities
    vx_nuc, vy_nuc, vz_nuc = coords[6], coords[7], coords[8]
    vx, vy, vz = coords[9], coords[10], coords[11]

    # Calculate Relative Distances
    r_nuc = np.sqrt(x_nuc**2.+y_nuc**2.+z_nuc**2.)              # distance between nucleus and host galaxy (the origin)
    r = np.sqrt(x**2.+y**2.+z**2.)                              # distance between particle and host galaxy (the origin)
    r1 = np.sqrt((x-x_nuc)**2. + (y-y_nuc)**2. + (z-z_nuc)**2.) # distance between nucleus and particle

    # Gravitational Accelerations

    # F = ma = GM(r)m/r^2
    # --> a = GM(r)/r^2
    # --> Where M(r) = k*r^(alpha)
    # --> a = G*k*r^(alpha)/r^2
    # --> a = G*k*r^(alpha - 2.)

    # Tuning Constants
    k = 15900.e11
    k1 = k /(25.)
    alpha = 1.

    # Overall Accelerations
    gr_nuc = G*k*r_nuc**(alpha-2.)
    gr = G*k*r**(alpha - 2.)
    gr1 = G*k1*r1**(alpha - 2.)

    # Convert Accelerations into Components
    gx_nuc, gy_nuc, gz_nuc = (-gr_nuc*x_nuc/r_nuc), (-gr_nuc*y_nuc/r_nuc), (-gr_nuc*z_nuc/r_nuc) # acceleration of nucleus due to host galaxy
    gx, gy, gz = (-gr*x/r), (-gr*y/r), (-gr*z/r)                                                 # acceleration of particle due to host galaxy
    gx1, gy1, gz1 = (-gr1*(x-x_nuc)/r1), (-gr1*(y-y_nuc)/r1), (-gr1*(z-z_nuc)/r1)                # acceleration of particle due to nucleus

    # Gather Velocities
    dxdt_nuc, dydt_nuc, dzdt_nuc = vx_nuc, vy_nuc, vz_nuc # velocity of nucleus (have initially)
    dxdt, dydt, dzdt = vx, vy, vz                         # velocity of particle (have initially)

    # Gather Accelerations
    dvxdt_nuc, dvydt_nuc, dvzdt_nuc = gx_nuc, gy_nuc, gz_nuc # depends on nucleus to host galaxy interaction/acceleration
    dvxdt, dvydt, dvzdt = gx+gx1, gy+gy1, gz+gz1             # depends on particle to host galaxy AND particle to nucleus interaction/acceleration

    # Pack into Single Array
    vels_accs = np.array([dxdt_nuc,dydt_nuc,dzdt_nuc,dxdt,dydt,dzdt,dvxdt_nuc,dvydt_nuc,dvzdt_nuc,dvxdt,dvydt,dvzdt])

    return vels_accs


def calc_vc(r) :
    '''
    Calculate speed of circular orbit 
    at a distance r from the center of a celestial body.
    All units in SI units.
    '''

    G = 6.673e-11
    k = 15900.e11
    alpha = 1.

    # F = mv^2/r = GM(r)m/r^2 (Centripetal equals Gravitational)
    # --> v^2 = GM(r)/r
    # --> Where M(r) = k*r^(alpha)
    # --> v^2 = G*k*r^(alpha - 1)
    # --> v = sqrt(G*k*r^(alpha-1))

    vc = np.sqrt(G*k*r**(alpha - 1.))

    return vc



####### INFALLING NUCLEUS SETUP ########

# Distance from origin to nucleus (placing along x-axis)
R_nuc = 254840.
x_nuc, y_nuc, z_nuc = R_nuc, 0., 0.

# Calculate orbital velocity if nucleus was in perfect circular orbit around host
v_circ = calc_vc(R_nuc)

# Perturb velocity to enforce a galaxy collision (no longer perfect circular orbit)
v_init_nuc = v_circ * .1

# Distribute velocity into components (placing all into y-component)
vx_nuc, vy_nuc, vz_nuc = 0., v_init_nuc, 0.

######### END OF NUCLEUS SETUP #########



###### INFALLING PARTICLES SETUP #######

N_part = 1000 # Total number of "particles" in galaxy (other than nucleus)

# Create a random spherical distribution of particles (in both physical / velocity space)
# Only one MPI rank needs to do this before we split up the data later
if (mpi_rank == 0):

    R = R_nuc / 20.0      # Maximum size of infalling galaxy
    R_v = v_init_nuc      # Maximum velocity of infalling particles (relative to nucleus)

    # Physical space
    phi = np.random.rand(N_part)*2.*np.pi       #  0 to 2pi
    costheta = (np.random.rand(N_part)*2.)-1.   # -1 to 1
    u = np.random.rand(N_part)                  #  0 to 1
    # Velocity space
    phi_v = np.random.rand(N_part)*2.*np.pi     #  0 to 2pi
    costheta_v = (np.random.rand(N_part)*2.)-1. # -1 to 1
    u_v = np.random.rand(N_part)                #  0 to 1

    # Assign the above random values to actual positions
    r = R * ( u**(1./3.)) # Uniformly distribute up to R
    theta = np.arccos( costheta )
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )

    # Assign the above random values to actual velocities
    r_v = R_v * ( u_v**(1./3.)) # Uniformly distribute up to R_v
    theta_v = np.arccos( costheta_v )
    vx = r_v * np.sin( theta_v ) * np.cos( phi_v )
    vy = r_v * np.sin( theta_v ) * np.sin( phi_v )
    vz = r_v * np.cos( theta_v )

    # Offset the particles based on nucleus position and velocity
    x = x + R_nuc
    vy = vy + v_init_nuc

else:
    # Initialize the above variables for other MPI tasks (needed for Scatter)
    x = None
    y = None
    z = None
    vx = None
    vy = None
    vz = None

# Initialize the variables for holding a subset of total particles
local_size = int(N_part/mpi_size)

x_local  = np.zeros(local_size)
y_local  = np.zeros(local_size)
z_local  = np.zeros(local_size)
vx_local = np.zeros(local_size)
vy_local = np.zeros(local_size)
vz_local = np.zeros(local_size)

# Scatter data uniformly across all MPI tasks
# Sends data from xyz to xyz_local from task 0
comm.Scatter(x, x_local, root=0)
comm.Scatter(y, y_local, root=0)
comm.Scatter(z, z_local, root=0)
comm.Scatter(vx, vx_local, root=0)
comm.Scatter(vy, vy_local, root=0)
comm.Scatter(vz, vz_local, root=0)

######## END OF PARTICLES SETUP ########



########### GENERATING DATA ############

# Create time array
tvec = np.linspace(0.,35000.,2200)
t_size = len(tvec)

# Create dummy data with correct shape
dummy_data = np.empty((N_part, t_size, 3))*0. # Shape: [Number of particles, Number of timesteps, Number of dimensions (x, y, z)]
dummy_nuc  = np.empty((1, t_size, 3))*0.

# Open and initialize HDF5 file with dummy data
f = h5py.File('galaxy.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
dset_pos_pt  = f.create_dataset("pos_pt",  data=dummy_data)
dset_vel_pt  = f.create_dataset("vel_pt",  data=dummy_data)
dset_pos_nuc = f.create_dataset("pos_nuc", data=dummy_nuc)
dset_vel_nuc = f.create_dataset("vel_nuc", data=dummy_nuc)

comm.Barrier()

# Each MPI taks loops over its subset of the particles
t1=tp.time()
for i in range(0, local_size):

    # Coordinates of particle
    init_coords = np.array([x_nuc,y_nuc,z_nuc,x_local[i],y_local[i],z_local[i],vx_nuc,vy_nuc,vz_nuc,vx_local[i],vy_local[i],vz_local[i]])

    # Integrate the ODE (simulate the particle)
    orbit = odeint(calc_vels_accs,init_coords,tvec)

    # Collect the data
    posx, posy, posz = orbit[:,3], orbit[:,4], orbit[:,5]
    velx, vely, velz = orbit[:,9], orbit[:,10], orbit[:,11]

    # Convert local particle index "i" to overall particle index
    index = ( mpi_rank * local_size ) + i

    # Write particle data to HDF5 file
    dset_pos_pt[index, :, 0] = posx
    dset_pos_pt[index, :, 1] = posy
    dset_pos_pt[index, :, 2] = posz
    dset_vel_pt[index, :, 0] = velx
    dset_vel_pt[index, :, 1] = vely
    dset_vel_pt[index, :, 2] = velz

t2 = tp.time()
print('MPI Rank %s : Simulating my particles took %s s' %(mpi_rank, t2-t1) )

# Collect data for nucleus
if (mpi_rank==0):
    posx_nuc, posy_nuc, posz_nuc = orbit[:,0], orbit[:,1], orbit[:,2]
    velx_nuc, vely_nuc, velz_nuc = orbit[:,6], orbit[:,7], orbit[:,8]

    # Write nucleus data to HDF5 file
    dset_pos_nuc[0, :, 0] = posx_nuc
    dset_pos_nuc[0, :, 1] = posy_nuc
    dset_pos_nuc[0, :, 2] = posz_nuc
    dset_vel_nuc[0, :, 0] = velx_nuc
    dset_vel_nuc[0, :, 1] = vely_nuc
    dset_vel_nuc[0, :, 2] = velz_nuc

comm.Barrier()

# Close HDF5 file
f.close()

######## END OF DATA GENERATION ########

if (mpi_rank==0):
    print('Success!')
