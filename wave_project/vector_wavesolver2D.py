from wavesolver2D import wavesolver2D
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib import animation

class vector_wavesolver2D(wavesolver2D):
    def __init__(self, b, Nx, Ny, Lx, Ly, T):
        super().__init__(b, Nx, Ny, Lx, Ly, T) #Inherit all variables and methods from wavesolver2D

    def update_ghost_cells(self, u):
        """
        Computes the ghost cells of the solution matrix u[i,j]
        """
        #Boundary at x = 0 and x = L_x
        u[0, 1:self.Ny+1] = u[2, 1:self.Ny+1]
        u[self.Nx+1, 1:self.Ny+1] = u[self.Nx-1, 1:self.Ny+1]

        #Boundary at y = 0 and y = L_y
        u[1:self.Nx+1, 0] = u[1:self.Nx+1, 2]
        u[1:self.Nx+1, self.Ny+1] = u[1:self.Nx+1, self.Ny-1]

    def set_conditions(self):
        """
        Sets up all necessary conditions to compute time evolution.
        """
        self.initiate_wave_velocity()
        self.initial_conditions()
        #Define certain constants
        self.A = 1./(2+self.b*self.dt)
        self.Cx = (self.dt/self.dx)**2
        self.Cy = (self.dt/self.dy)**2

    def initiate_wave_velocity(self):
        """
        Initiates a matrix that stores the wave velocity at all points in the mesh
        """

        for i in range(1, self.Nx+1):
            for j in range(1, self.Ny+1):
                self.q[i,j] = self.q_func(self.x[i-1], self.y[j-1])
        self.c = np.sqrt(np.max(self.q))
        self.dt = self.beta*1./np.sqrt( (1/self.dx**2) + (1/self.dy**2) )*(1./self.c)
        #define ghost points
        self.q[0, 1:self.Ny+1] = 2*self.q[1, 1:self.Ny+1] - self.q[2, 1:self.Ny+1]
        self.q[self.Nx+1, 1:self.Ny+1] = 2*self.q[self.Nx, 1:self.Ny+1] - self.q[self.Nx-1, 1:self.Ny+1]

        self.q[1:self.Nx+1, 0] = 2*self.q[1:self.Nx+1, 1] - self.q[1:self.Nx+1, 2]
        self.q[1:self.Nx+1, self.Ny+1] = 2*self.q[1:self.Nx+1, self.Ny] - self.q[1:self.Nx+1, self.Ny-1]

    def initial_conditions(self):
        """
        Imposes the initial condition u(x,y,0) = I(x,y).
        """
        self.u_old[1:self.Nx+1, 1:self.Ny+1] = self.I(self.X, self.Y)
        self.update_ghost_cells(self.u_old)



    def compute_first_step(self):
        """
        Special formula for the first step
        """
        self.u[1:self.Nx+1, 1:self.Ny+1] = 0.25*( 2*self.dt*(2-self.b*self.dt)*self.V(self.X, self.Y) + 4*self.u_old[1:self.Nx+1, 1:self.Ny+1] \
        + self.Cx*( (self.q[2:self.Nx+2, 1:self.Ny+1] + self.q[1:self.Nx+1, 1:self.Ny+1])*(self.u_old[2:self.Nx+2, 1:self.Ny+1] - self.u_old[1:self.Nx+1, 1:self.Ny+1]) \
        - (self.q[1:self.Nx+1, 1:self.Ny+1] + self.q[:self.Nx, 1:self.Ny+1])*(self.u_old[1:self.Nx+1, 1:self.Ny+1] - self.u_old[:self.Nx, 1:self.Ny+1]) ) \
        + self.Cy*( (self.q[1:self.Nx+1, 2:self.Ny+2] + self.q[1:self.Nx+1, 1:self.Ny+1])*(self.u_old[1:self.Nx+1, 2:self.Ny+2] - self.u_old[1:self.Nx+1, 1:self.Ny+1]) \
        - (self.q[1:self.Nx+1, 1:self.Ny+1] + self.q[1:self.Nx+1, :self.Ny])*(self.u_old[1:self.Nx+1, 1:self.Ny+1] - self.u_old[1:self.Nx+1, :self.Ny]) ) \
        + 2*self.dt**2*self.f(self.X, self.Y, 0.) )
        self.update_ghost_cells(self.u)

    def advance(self):
        """
        Advances one step forward in time.
        """
        Nx = self.Nx
        Ny = self.Ny

        self.u_new[1:Nx+1, 1:Ny+1] = self.A*( (self.b*self.dt-2)*self.u_old[1:Nx+1, 1:Ny+1] + 4*self.u[1:Nx+1, 1:Ny+1] \
        + self.Cx*( (self.q[2:Nx+2, 1:Ny+1] + self.q[1:Nx+1, 1:Ny+1])*(self.u[2:Nx+2, 1:Ny+1] - self.u[1:Nx+1, 1:Ny+1]) - (self.q[1:Nx+1, 1:Ny+1] + self.q[:Nx, 1:Ny+1])*(self.u[1:Nx+1, 1:Ny+1] - self.u[:Nx, 1:Ny+1]) ) \
        + self.Cy*( (self.q[1:Nx+1, 2:Ny+2] + self.q[1:Nx+1, 1:Ny+1])*(self.u[1:Nx+1, 2:Ny+2] - self.u[1:Nx+1, 1:Ny+1]) - (self.q[1:Nx+1, 1:Ny+1] + self.q[1:Nx+1, :Ny])*(self.u[1:Nx+1, 1:Ny+1] - self.u[1:Nx+1, :Ny]) ) \
        + 2*self.dt**2*self.f(self.X, self.Y, self.t) )

    def solve(self):
        """
        Computes the time evolution of the 2D wave equation.
        """
        self.compute_first_step()
        start = time()
        self.t = self.dt
        while self.t <= self.T:
            self.advance()
            self.update_ghost_cells(self.u_new)
            self.swap()
            self.t += self.dt
        end = time()
        timeused = end-start



    def compute_error(self, analytical):
        error_mesh_func = np.abs(self.u[1:-1, 1:-1] - analytical(self.X,self.Y,self.t))
        self.linf_norm = np.max(error_mesh_func)
        return self.linf_norm, self.dx
