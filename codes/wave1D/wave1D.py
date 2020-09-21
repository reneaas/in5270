import numpy as np
import matplotlib.pyplot as plt

class wave1D:
    def __init__(self, h, L, c, T):
        self.C = 0.75
        self.c = c
        self.T = T
        self.dt = h
        self.dx = self.c*self.dt/self.C
        self.L = L
        self.Nx = int(round(self.L/self.dx))
        self.x = np.linspace(0, self.L, self.Nx)
        self.CC = self.C**2

        self.u_new = np.zeros(self.Nx)
        self.u = np.zeros(self.Nx)
        self.u_old = np.zeros(self.Nx)
        self.linf_norm = 0.

    def set_conditions(self, I):
        self.u_old[:] = I(self.x)
        #self.u_old[0] = 0.; self.u_old[-1] = 0.;
        self.u[1:-1] = self.u_old[1:-1] + 0.5*self.CC*(self.u_old[2:] - 2*self.u_old[1:-1] + self.u_old[:-2])
        self.u[0] = 0.
        self.u[-1] = 0.

    def compute_next_step(self):
        self.u_new[1:-1] = 2*self.u[1:-1] - self.u_old[1:-1] + self.CC*(self.u[2:] - 2*self.u[1:-1] + self.u[:-2])

        tmp = self.u_old
        self.u_old = self.u
        self.u = self.u_new
        self.u_new = tmp


    def compute_time_evolution(self, analytical):
        self.t = self.dt
        while self.t <= self.T:
            self.compute_next_step()
            error_mesh_func = analytical(self.x[1:-1], self.t) - self.u_new[1:-1]
            self.linf_norm = max(self.linf_norm, np.abs(error_mesh_func).max())
            self.t += self.dt
        return self.linf_norm

    def compute_error(self, analytical):
        error_mesh_func = analytical(self.x, self.T) - self.u[:]
        self.linf_norm = max(self.linf_norm, np.abs(error_mesh_func).max())
        return self.linf_norm
