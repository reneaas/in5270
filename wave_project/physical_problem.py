from vector_wavesolver2D import vector_wavesolver2D
from scalar_wavesolver2D import scalar_wavesolver2D
import numpy as np
import matplotlib.pyplot as plt

def I(x,y, I0 = 1, Ia = 1., Im = 0.5, Is = 0.1):
    return I0 + Ia*np.exp(-((x-Im)/Is)**2)

def V(x,y):
    return 0.

def f(x,y,t):
    return 0.

def q(x,y, g=9.81, H0 = 10):
    return g*(H0 - B(x,y))

def B(x,y, B0=0., Ba = 1., Bmx = 0.5, Bmy = 0.5, Bs = 0.1):
    return B0 + Ba*np.exp(-( ((x-Bmx)/Bs)**2 + ((y-Bmy)/Bs)**2 ) )


Nx = 50
Ny = 50
Lx = 1.
Ly = 1.
b = 0.
T = 10.

times = [0.005*i for i in range(10)]
for T in times:
    my_solver = vector_wavesolver2D(b=b, Nx = Nx, Ny = Ny, Lx = Lx, Ly = Ly, T = T)
    my_solver.set_function_conditions(I = I, V = V, f = f, q = q)
    my_solver.set_conditions()
    my_solver.solve()
    my_solver.plot_solution(analytical=None)
