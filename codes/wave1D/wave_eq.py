import numpy as np
#from wavesolver1D import wavesolver1D
from wave1D import wave1D
import matplotlib.pyplot as plt

L = 1. #Length scale
c = 1. #Wave velocity
T = 1. #Total time

def I(x):
    """
    Initial condition u(x,0) = I(x).
    """
    return np.sin(2*np.pi*x/L)


def analytical(x,t):
    return np.sin(2*np.pi*x/L)*np.cos(2*np.pi*c*t/L)

h0 = 0.01
stepsizes = [h0/2**i for i in range(6)]
l2_norms = []
linf_norms = []

for h in stepsizes:
    print("running for h = ", h)
    my_solver = wave1D(h, L, c, T)
    my_solver.set_conditions(I)
    linf_norm = my_solver.compute_time_evolution(analytical)
    linf_norms.append(linf_norm)

plt.plot(np.log(stepsizes), np.log(linf_norms))
plt.show()

r = []
for i in range(1, len(linf_norms)):
    r.append(np.log(linf_norms[i]/linf_norms[i-1]) / np.log(stepsizes[i]/stepsizes[i-1]))

print(r)
