#!/usr/bin/python
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import math
import datetime
import scipy.signal
import time

start = time.time()



# Simulation Parameters`

dt = 0.01  # Temporal seperation

# Spatial grid points
nx = 21
ny = 21

Lx = 12  # Spatial size, symmetric with respect to x=0
Ly = 12  # Spatial size, symmetric with respect to x=0

alpha=1

kx = 1
ky = 1

# Derived Simulation Parameters

dx = Lx/(nx-1) # Spatial seperation
dy = Ly/(ny-1) # Spatial seperation

x = np.linspace(-0.5*Lx, 0.5*Lx, nx)
y = np.linspace(-0.5*Ly, 0.5*Ly, ny)


# This function takes size (n) and seperation (dx) as Parameters
# and produces a discrete 1D Laplacian matrix.
def laplacian1D(size, separation):
    result = np.zeros((size, size))
    for i in range(size):
        result[i][i]=-2
    for i in range(size-1):
        result[i][i+1]=1
        result[i+1][i]=1
    return 1/(separation**2) * result

# This function creates a discrete 2D Laplacian matrix by taking kronecker sum
# of two discrete 1D Laplacian matrix.
def laplacian2D():
    Dxx = laplacian1D(nx, dx)
    Dyy = laplacian1D(ny, dy)

    return sp.kron(sp.eye(ny), Dxx) + sp.kron(Dyy,sp.eye(nx))

# This function calculates the anisotropic potential energy at specified position.
# alpha is defined in such a way that force constant omega_y^2 = alpha * omega_x^2   .
def Potential(posx, posy):
    return 0.5*(posx**2) + 0.5*(alpha * (posy**2))

# This function creates a discrete 2D potential energy matrix. It's compatible
# with flattening with row priority.
def VMatrix():
    result = np.zeros(( x.size*y.size , x.size*y.size ))
    cursor = 0
    for i in range(y.size):
        for j in range(x.size):
            result[cursor][cursor] = Potential(x[j] , y[i])
            cursor += 1

    return result





# This function creates a discrete 2D kinetic energy matrix.
def TMatrix():
    return -0.5 * laplacian2D().toarray()

# This function creates a discrete 2D Hamiltonian matrix.
def HMatrix():
    return VMatrix() + TMatrix()

def SquareSum2D(some_psi):
    sum = 0
    for i in range(len(some_psi)):
        for j in range(len(some_psi[i])):
            sum += np.abs(some_psi[i][j])**2 * dx * dy

    return sum


# This function normalizes the given 2D wave.
def Normalize2D(some_psi):
    scale_factor = np.sqrt(1/SquareSum2D(some_psi))
    return scale_factor * some_psi


# This function finds the specified order of excited state with respect to
# given Hamiltonian. (order = 0 for groundstate, order = 1 for first excited
# state vice versa.)
def ExcitedStateNumerical2D(order):
    H = HMatrix()
    val,vec=la.eig(H)
    z = np.argsort(val)

    psi = vec[:,z[order]]

    return Normalize2D(psi.reshape(ny, nx))



# This function returns the analytical solution to the ground state
def GroundStateExact2D():
    result = np.zeros((ny, nx))

    for i in range(ny):
        for j in range(nx):
            currentx = x[j]
            currenty = y[i]
            psi = (alpha**(0.125))*(np.exp(-0.5*(currentx**2)))*((np.exp(-0.5*(alpha**0.5)*(currenty**2))))
            result[i][j] = psi

    return Normalize2D(result)


# Name says it all
def IdentityMatrix():
    I = np.zeros((x.size*y.size, x.size*y.size))
    for i in range(x.size*y.size):
        I[i][i] = 1
    return I

def LHS():
    first_term = IdentityMatrix()
    second_term = (1j*HMatrix()*dt)/(2)

    return first_term + second_term

def RHS():
    first_term = IdentityMatrix()
    second_term = (1j*HMatrix()*dt)/(2)

    return first_term - second_term




LHS = LHS()
RHS = RHS()
InverseOfLHS = np.linalg.inv(LHS)

# U is the time evolution matrix. To keep things time efficient,
# I initialize this matrix once and use it as necessary.
U = np.matmul(InverseOfLHS, RHS)



# This function applies time evolution to a given wavefunction.
def Evolve(some_psi):
    return np.matmul(U, np.ndarray.flatten(some_psi)).reshape(ny,nx)


# This function calculates the overlap between two 2D wavefunctions
def Overlap2D(psi1, psi2):
    overlap = 0
    for i in range(x.size):
        for j in range(y.size):
            overlap += psi1[i][j] * np.conj(psi2[i][j]) * dx * dy
    return np.abs(overlap)

# This function calculates the overlap between two 1D wavefunctions
def Overlap1D(psi1, psi2):
    overlap = 0
    for i in range(x.size):
        overlap += psi1[i] * np.conj(psi2[i]) * dx
    return np.abs(overlap)




def CoherentStateNumerical2D():
    startingPoint = CoherentStateExact2D(0)
    return startingPoint


# Implementation of coherent state specified in the
#
# https://files.slack.com/files-pri/T5MM8M0CR-F02LPDYN02X/download/2021-11-09-2d_coherent_state.pdf
#
# with x_0=2
def CoherentStateExact2D(t):
    xinitial = 3
    result = np.zeros((ny, nx), dtype=complex)
    for i in range(ny):
        for j in range(nx):
            currentx = x[j]
            currenty = y[i]
            #A = (alpha**0.125)/(np.pi**0.5)
            X = np.exp(  -0.5 * (currentx - xinitial * np.cos(t))**2    ) * np.exp(  -1j * xinitial * currentx * np.sin(t))
            Y = np.exp(  -(0.5 * (alpha**0.5)) * (currenty - ky*np.sin(t*(alpha**0.5)))**2    ) * np.exp(  1j * ky * currenty * np.cos(t**(alpha**0.5)))
            result[i][j] = X*Y

    return Normalize2D(result)





number_of_oscillations = 5

terminateAt = number_of_oscillations * 2 * np.pi
timesteps = math.ceil(terminateAt / dt)

currentDate  = datetime.datetime.now()

lx_over_kx = Lx/kx
ly_over_ky = Ly/ky

filename = 'default.dat'

with open(filename, 'a') as f:
    f.write('# Simulation started at ' + str(currentDate) + '\n')
    f.write('#\n')
    f.write('# Simulation details: \n')
    f.write('# Lx = ' + str(Lx) + ' \n')
    f.write('# kx = ' + str(kx) + ' \n')
    f.write('# Lx / kx = ' + str(lx_over_kx) + ' \n')

    f.write('# Ly = ' + str(Ly) + ' \n')
    f.write('# ky = ' + str(ky) + ' \n')
    f.write('# Ly / ky = ' + str(ly_over_ky) + ' \n')
    f.write('# dt = ' + str(dt) + ' s \n')
    f.write('# dx = ' + str(dx) + ' \n')
    f.write('# dy = ' + str(dy) + ' \n')
    f.write('# Spatial grid points : ' + str(nx) + ' x ' + str(ny) + ' \n')
    f.write('#\n')
    f.write('# time\terror\n')

psi_num = CoherentStateNumerical2D()

for i in range(timesteps):
    currentTime = i*dt

    psi_exact = CoherentStateExact2D(currentTime)
    error = np.abs(1 - Overlap2D(psi_num, psi_exact))

    current_time_as_string = "{:.3f}".format(currentTime)
    print('Current time: ' + current_time_as_string)
    error_as_string = "{:.4e}".format(error)

    with open(filename, 'a') as f:
        f.write(current_time_as_string+'\t'+error_as_string+'\n')


    '''
    plt.suptitle('Time: ' + current_time_as_string + '\nError: ' + error_as_string)

    plt.subplot(1, 2, 1)
    plt.contourf(x, y, np.abs(psi_num)**2, 256, cmap='RdGy')
    plt.title('Numerical solution')
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.contourf(x, y, np.abs(psi_exact)**2, 256, cmap='RdGy')
    plt.title('Exact solution')
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.grid()

    plt.tight_layout()
    plt.savefig('frame'+str(i)+'.png')
    plt.clf()
    '''

    psi_num = Evolve(psi_num)


end = time.time()
difference = int(end - start)
print( str(difference) + 's')
