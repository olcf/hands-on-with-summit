# data_transfer.py
# Author: Michael A. Sandoval
import numpy as np
import cupy as cp

# Initialize the array x_gpu_0 on GPU 0
x_gpu_0 = cp.arange(10)
cp.cuda.Stream.null.synchronize() # Waits for GPU 0 to finish
print(x_gpu_0.device, 'done')

# Modify a copy of x_gpu_0 on GPU 1 (must send data to GPU 1)
with cp.cuda.Device(1):
        x_gpu_1 = x_gpu_0 # TO-DO
        x_gpu_1 = x_gpu_1**2
        cp.cuda.Stream.null.synchronize() # Waits for GPU 1 to finish
        print(x_gpu_1.device, 'done')

# Modify a copy of x_gpu_0 on GPU 2 (must send data to GPU 2)
with cp.cuda.Device(2):
        x_gpu_2 = x_gpu_0 # TO-DO
        x_gpu_2 = x_gpu_2**3
        cp.cuda.Stream.null.synchronize() # Waits for GPU 2 to finish
        print(x_gpu_2.device, 'done')

# Sum all arrays on the CPU (must send arrays to the CPU)
x_cpu = x_gpu_0 +  x_gpu_1 + x_gpu_2 # TO-DO
print('Finished computing on the CPU\n')

# Summary of our results
print('Results:')
print(x_gpu_0.device, ':', x_gpu_0)
print(x_gpu_1.device, ':', x_gpu_1)
print(x_gpu_2.device, ':', x_gpu_2)
print('CPU: ', x_cpu)

# Check results
solution = np.array((0,3,14,39,84,155,258,399,584,819))
if ( (x_cpu==solution).all() ):
    print('Success!')
else:
    print('Something went wrong, try again!')
