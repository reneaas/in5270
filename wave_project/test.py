#from wavesolver2D import wavesolver2D
from scalar_wavesolver2D import scalar_wavesolver2D
from vector_wavesolver2D import vector_wavesolver2D

def q(x,y):
    return 1.

def I(x,y):
    U = 1.
    return U

def f(x,y,t):
    return 0

def V(x,y):
    return 0


Nx = 100
Ny = 100
T = 100
my_solver = scalar_wavesolver2D(b=0., Nx = Nx, Ny=Ny, Lx = Nx-1, Ly = Ny-1, T = T)
my_solver.set_function_conditions(I = I, V = V, f = f, q = q)
my_solver.set_conditions()
my_solver.solve()

my_solver = vector_wavesolver2D(b=0., Nx = Nx, Ny=Ny, Lx = Nx-1, Ly = Ny-1, T = T)
my_solver.set_function_conditions(I = I, V = V, f = f, q = q)
my_solver.set_conditions()
my_solver.solve()
