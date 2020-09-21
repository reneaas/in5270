import numpy as np
import matplotlib.pyplot as plt
from time import time
from matplotlib import animation


class wavesolver2D:
    def __init__(self, b, Nx, Ny, Lx, Ly, T):
        self.b = b  #Damping coefficient
        self.Nx = Nx #Points in the x-direction
        self.Ny = Ny #Points in the y-direction
        self.Lx = Lx #Length in x-direction
        self.Ly = Ly #Length in y-direction
        self.T = T #Total simulation time
        self.beta = 0.9


        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.q = np.zeros((self.Nx + 2, self.Ny + 2))
        self.u = np.zeros((self.Nx + 2, self.Ny + 2))
        self.u_new = np.zeros((self.Nx + 2, self.Ny + 2))
        self.u_old = np.zeros((self.Nx + 2, self.Ny + 2))
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.X = np.transpose(self.X)
        self.Y = np.transpose(self.Y)


    def swap(self):
        self.u_old, self.u, self.u_new = self.u, self.u_new, self.u_old

    def set_function_conditions(self, I, V, f, q):
        self.I = lambda x, y: I(x,y) #Initial condition
        self.V = lambda x, y: V(x,y) #Initial velocity
        self.f = lambda x, y, t: f(x,y,t) #Driving force
        self.q_func = lambda x, y: q(x,y) #Variable wave velocity

    def print_solution(self):
        print("u = ", self.u[1:self.Nx+1])

    def plot_solution(self, analytical):
        if analytical == None:
            #X, Y = np.meshgrid(self.x, self.y)
            plt.contourf(self.X, self.Y, self.u[1:-1, 1:-1], cmap="inferno")
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Approximation")
            #plt.legend()
            plt.show()
        else:
            #X, Y = np.meshgrid(self.x, self.y)
            plt.contourf(self.X,self.Y,self.u[1:-1, 1:-1], cmap="inferno")
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Approximation")
            #plt.legend()
            plt.figure()

            plt.contourf(self.X,self.Y,analytical(self.X,self.Y, self.t), cmap="inferno")
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Exact")
            #plt.legend()
            plt.show()
