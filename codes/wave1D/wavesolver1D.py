import numpy as np
import matplotlib.pyplot as plt

class wavesolver1D:
    def __init__(self, h, L, c, T):
        self.C = 0.75             #Courant number
        self.L = L
        self.c = c
        self.T = T
        self.dt = h
        self.dx = self.c*self.dt/self.C
        self.Nx = int(round(self.L/self.dx))
        self.Nt = int(round(self.T/self.dt))
        self.x = np.linspace(0, L, self.Nx)
        self.CC = self.C*self.C

        self.u_new = np.zeros(self.Nx)
        self.u = np.zeros(self.Nx)
        self.u_old = np.zeros(self.Nx)
        self.l2_norm = 0.
        self.linf_norm = 0.

    def Initialize(self, I):
        self.u_old = I(self.x)
        self.u_old[0] = 0.
        self.u_old[-1] = 0.
        self.u[0] = 0.
        self.u[-1] = 0.
        self.u_new[0] = 0.
        self.u_new[-1] = 0.

    def FirstStep(self):
        for i in range(1,self.Nx-1):
            self.u[i] = self.u_old[i] - 0.5*self.CC*(self.u_old[i+1] - 2*self.u_old[i] + self.u_old[i-1])

    def FirstStep_vec(self):
        self.u[1:-1] = self.u_old[1:-1] - 0.5*self.CC*(self.u_old[2:] - 2*self.u_old[1:-1] + self.u_old[0:-2])

    def NextStep_vec(self):
        self.u_new[1:-1] = 2*self.u[1:-1] - self.u_old[1:-1] + self.CC*(self.u[2:] - 2*self.u[1:-1] + self.u[0:-2])


    def NextStep(self):
        for i in range(1, self.Nx-1):
            self.u_new[i] = 2*self.u[i] - self.u_old[i] + self.CC*( self.u[i+1] - 2*self.u[i] + self.u[i-1] )


    def Swap(self):
        self.u_old[:] = self.u
        self.u[:] = self.u_new
        self.u[0] = 0.; self.u[-1] = 0.;
        self.u_old[0] = 0.; self.u_old[-1] = 0.;
        self.u_new[0] = 0.; self.u_new[-1] = 0.;

    def ComputeTimeEvolution(self, analytical):
        self.t = 0.
        self.FirstStep_vec()
        self.t += self.dt
        while self.t <= self.T:
            self.NextStep_vec()
            self.Swap()
            #self.t += self.dt
            self.ComputeTotalError(analytical)
            self.t += self.dt
        self.l2_norm = np.sqrt(self.dt*self.dx*self.l2_norm)
        return self.l2_norm, self.linf_norm

    def PlotSolution(self, analytical):
        analytical_vals = analytical(self.x, self.T)
        plt.plot(self.x, self.u, label = "Numerical solution")
        plt.plot(self.x, analytical_vals, label = "Analytical solution")
        plt.xlabel("x")
        plt.ylabel("u(x,T)")
        plt.title("Wave solution at t = " + str(self.t))
        plt.legend()
        plt.show()


    def ComputeTotalError(self, analytical):
        error_mesh_func = analytical(self.x, self.t) - self.u[:]
        trial_error = np.abs(error_mesh_func).max()
        self.linf_norm = max(trial_error, self.linf_norm)
        self.l2_norm += np.sum(error_mesh_func**2)
