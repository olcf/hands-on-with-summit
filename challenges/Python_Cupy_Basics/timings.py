# timings.py
# Author: Michael A. Sandoval
import cupy as cp
import numpy as np
import time as tp

A      = np.random.rand(3000,3000) # NumPy rand
G      = cp.random.rand(3000,3000) # CuPy rand
G32    = cp.random.rand(3000,3000,dtype=cp.float32) # Create float32 matrix instead of float64 (default)
G32_9k = cp.random.rand(9000,1000,dtype=cp.float32) # Create float32 matrix of a different shape

t1 = tp.time()
np.linalg.svd(A) # NumPy Singular Value Decomposition
t2 = tp.time()
print("CPU time: ", t2-t1)

t3 = tp.time()
cp.linalg.svd(G) # CuPy Singular Value Decomposition
cp.cuda.Stream.null.synchronize() # Waits for GPU to finish
t4 = tp.time()
print("GPU time: ", t4-t3)

t5 = tp.time()
cp.linalg.svd(G32)
cp.cuda.Stream.null.synchronize()
t6 = tp.time()
print("GPU float32 time: ", t6-t5)

t7 = tp.time()
cp.linalg.svd(G32_9k)
cp.cuda.Stream.null.synchronize()
t8 = tp.time()
print("GPU float32 restructured time: ", t8-t7)
