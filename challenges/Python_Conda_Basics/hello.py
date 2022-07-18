# hello.py
# Author: Michael A. Sandoval
import platform
import numpy

py_vers = platform.python_version()
np_vers = numpy.__version__

print("Hello from Python %s!"  %(py_vers) )
print("You are using NumPy %s" %(np_vers) )
