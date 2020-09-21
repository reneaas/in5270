#from wavesolver2D import wavesolver2D
from scalar_wavesolver2D import scalar_wavesolver2D
from vector_wavesolver2D import vector_wavesolver2D
import numpy as np

def q(x,y):
    return y

def I(x,y,nx=1, my=1, Lx = 1, Ly = 1):
    kx = nx*np.pi/Lx
    ky = my*np.pi/Ly
    return np.cos(kx*x)*np.cos(ky*y)

def f(x, y, t, nx=1, my=1, omega=1., d=1., Lx = 1, Ly = 1):
    kx = nx*np.pi/Lx
    ky = my*np.pi/Ly
    k_squared = kx**2 + ky**2
    u = np.cos(omega*t)*np.exp(-d*t)*np.cos(kx*x)*np.cos(ky*y)
    return (k_squared*y + ky*np.tan(ky*y)-2*omega**2)*u

"""
def f(x, y, t, nx=1, my=1, omega=1., d=0.01, Lx = 1, Ly = 1, b=0.01):
    kx = nx*np.pi/Lx
    ky = my*np.pi/Ly
    k_squared = kx**2 + ky**2
    u = np.cos(omega*t)*np.exp(-d*t)*np.cos(kx*x)*np.cos(ky*y)
    return ((k_squared*y + ky*np.tan(ky*y) + d**2 - omega**2 - b*d) - (2*d*omega - omega*b)*np.tan(omega*t))*u
"""

def V(x, y, nx=1, my=1, Lx = 1, Ly = 1, d=1):
    kx = nx*np.pi/Lx
    ky = my*np.pi/Ly
    return -d*np.cos(kx*x)*np.cos(ky*y)

def analytical(x, y, t, nx=1, my=1, omega=1., d=1, Lx = 1, Ly = 1):
    kx = nx*np.pi/Lx
    ky = my*np.pi/Ly
    k_squared = kx**2 + ky**2
    return np.cos(omega*t)*np.exp(-d*t)*np.cos(kx*x)*np.cos(ky*y)



Lx = 1
Ly = 1
T = 1
#Compute convergence rate:
r = []
E = []
h = []
N = [2**i for i in range(1, 8)]
print(N)
for n in N:
    Nx = n
    Ny = n
    print("n = ", n)
    my_solver = vector_wavesolver2D(b=2., Nx = Nx, Ny=Ny, Lx = Lx, Ly = Ly, T = T)
    my_solver.set_function_conditions(I = I, V = V, f = f, q = q)
    my_solver.set_conditions()
    my_solver.solve()
    #my_solver.plot_solution(analytical)
    #linf_norm, dx = my_solver.compute_error(analytical)
    linf_norm, dx = my_solver.compute_error(analytical)
    #my_solver.plot_solution(analytical)
    E.append(linf_norm)
    h.append(dx)
my_solver.plot_solution(analytical)

for i in range(len(E)-1):
    r.append( np.log10(E[i+1]/E[i])/np.log10(h[i+1]/h[i]) )
print(r)
