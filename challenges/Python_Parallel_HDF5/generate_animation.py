# generate_animation.py
# Author: Michael A. Sandoval
# Generates an animation / GIF of galaxy.py output

import matplotlib.pyplot as plt
import numpy as np
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Update plots function (for animation)
def update(num, data, line, line2):
    particles_now = particles[:,num,:]/1000.
    line.set_data(particles_now[:,0], particles_now[:,1])
    line.set_3d_properties(particles_now[:,2])

    nucleus_now = nucleus[:,num,:]/1000.
    line2.set_data(nucleus_now[:,0], nucleus_now[:,1])
    line2.set_3d_properties(nucleus_now[:,2])

# Read in data
f=h5py.File('galaxy.hdf5', 'r')
particles = f['pos_pt'][:] # n , time, dim
nucleus = f['pos_nuc'][:] # 0, time, dim
f.close()

particles_now = particles[:,0,:]/1000. # convert to km
nucleus_now = nucleus[:,0,:]/1000. # convert to km

# Initialize figure/axes
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot "host galaxy"
ax.plot(0,0,0,'o', markersize=10, alpha=0.5)

# Plot infalling particles
line, = ax.plot(particles_now[:, 0], particles_now[:, 1], particles_now[:, 2], 'ko', markersize=1)

# Plot infalling nucleus
line2, = ax.plot(nucleus_now[:,0], nucleus_now[:,1], nucleus_now[:,2], 'mo', markersize=3)

# Setting the axes properties
ax.set_xlim3d([-250.0, 250.0])
ax.set_xlabel('X [km]')

ax.set_ylim3d([-250.0, 250.0])
ax.set_ylabel('Y [km]')

ax.set_zlim3d([-250.0, 250.0])
ax.set_zlabel('Z [km]')

# Set initial 3D view
ax.view_init(elev=110., azim=-90.)

# Save animation
ani = animation.FuncAnimation(fig, update, frames=1000, fargs=(particles_now, line, line2), interval=50, blit=False)
ani.save('galaxy_collision.gif', writer='imagemagick')
