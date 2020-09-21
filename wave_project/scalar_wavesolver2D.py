from wavesolver2D import wavesolver2D
import numpy as np
from time import time

class scalar_wavesolver2D(wavesolver2D):
    def __init__(self, b, Nx, Ny, Lx, Ly, T):
        super().__init__(b, Nx, Ny, Lx, Ly, T) #Inherit all variables and methods from wavesolver2D.


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
        for j in range(1, self.Ny+1):
            self.q[0, j] = 2*self.q[1, j] - self.q[2, j]
            self.q[self.Nx+1, j] = 2*self.q[self.Nx, j] - self.q[self.Nx-1, j]

        for i in range(1, self.Nx+1):
            self.q[i, 0] = 2*self.q[i, 1] - self.q[i, 2]
            self.q[i, self.Ny+1] = 2*self.q[i, self.Ny] - self.q[i, self.Ny-1]


    def initial_conditions(self):
        """
        Imposes the initial condition u(x,y,0) = I(x,y).
        """
        for i in range(1, self.Nx+1):
            for j in range(1, self.Ny+1):
                self.u_old[i,j] = self.I(self.x[i-1], self.y[j-1])
        self.update_ghost_cells(self.u_old)

    def update_ghost_cells(self, u):
        """
        Computes ghost cells for the solution matrix u[i,j].
        """
        #Boundary at x = 0 and x = L_x
        for j in range(1, self.Ny+1):
            u[0, j] = u[2, j]
            u[self.Nx+1, j] = u[self.Nx-1, j]

        #Boundary at y = 0 and y = L_x
        for i in range(1, self.Nx+1):
            u[i, 0] = u[i, 2]
            u[i, self.Ny+1] = u[i, self.Ny-1]

    def forward_x(self, q, u, i, j):
        return (q[i+1,j]+q[i,j])*(u[i+1,j]-u[i,j])

    def backward_x(self, q, u, i, j):
        return (q[i,j]+q[i-1,j])*(u[i,j]-u[i-1,j])

    def forward_y(self, q, u, i, j):
        return (q[i,j+1]+q[i,j])*(u[i,j+1]-u[i,j])

    def backward_y(self, q, u, i, j):
        return (q[i,j]+q[i,j-1])*(u[i,j]-u[i,j-1])

    def compute_first_step(self):
        for i in range(1, self.Nx+1):
            for j in range(1, self.Ny+1):
                self.u[i,j] = 0.25*( 2*self.dt*(2-self.b*self.dt)*self.V(self.x[i-1], self.y[j-1]) + 4*self.u_old[i,j] + self.Cx*( self.forward_x(self.q, self.u_old, i,j) - self.backward_x(self.q, self.u_old, i, j) )  + self.Cy*( self.forward_y(self.q, self.u_old, i,j) - self.backward_y(self.q, self.u_old, i, j) ) + 2*self.dt**2*self.f(self.x[i-1], self.y[j-1], 0.) )
        self.update_ghost_cells(self.u)

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
        print("Timeused = ", timeused)

    def compute_error(self, analytical):
        self.linf_norm = 0.
        X, Y = np.meshgrid(self.x, self.y)
        error_mesh_func = self.u[1:-1,1:-1]-analytical(X, Y, self.t)
        for i in range(self.Nx):
            self.linf_norm = max(np.max(error_mesh_func[i]), self.linf_norm)
        return self.linf_norm, self.dx

    def advance(self):
        for i in range(1, self.Nx+1):
            for j in range(1, self.Ny+1):
                self.u_new[i,j] = self.A*( (self.b*self.dt-2)*self.u_old[i,j] + 4*self.u[i,j] + self.Cx*( self.forward_x(self.q, self.u, i, j) - self.backward_x(self.q, self.u, i, j) ) + self.Cy*( self.forward_y(self.q, self.u, i, j) - self.backward_y(self.q, self.u, i, j) ) + 2*self.dt**2*self.f(self.x[i-1], self.y[j-1], self.t) )
