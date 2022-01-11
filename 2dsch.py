#!/usr/bin/python
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import math
import datetime
import time

start = time.time()


# Simulation Parameters

dt = 0.01  # Temporal seperation

# Spatial grid points
nx = 121
ny = 121

Lx = 11  # Spatial size, symmetric with respect to x=0
Ly = 8.8  # Spatial size, symmetric with respect to x=0

kappa=4 # Anisotropy constant


x0 = 2
ky = 1

filename = 'xinitial' + str(x0) + 'ky' + str(ky) + '.dat'
#filename = 'xinitial'+str(x0)+'+ky' + str(ky) +'.dat'
number_of_oscillations = 20

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
# kappa is defined in such a way that force constant omega_y^2 = kappa * omega_x^2   .
def Potential(posx, posy):
    return 0.5*(posx**2) + 0.5*(kappa * (posy**2))

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

# This function computes the sum of squares of each element of a 2D matrix.
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

            normalization_const = (kappa**(0.125))/(np.pi**0.5)

            x_part = np.exp(-0.5 * currentx**2)

            y_part = np.exp((kappa**0.5) * -0.5 * currenty**2)

            result[i][j] = normalization_const * x_part * y_part

    return result



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


def InitializeU():
    ustart = time.time()
    LHSmatrix = LHS()
    RHSmatrix = RHS()
    InverseOfLHS = np.linalg.inv(LHSmatrix)
    ufinish = time.time()
    elapsed = int(ufinish - ustart)
    print( 'Calculation of U (' + str(nx) +' x '+ str(ny) + ') took ' + str(elapsed) + 's')
    return np.matmul(InverseOfLHS, RHSmatrix)

# U is the time evolution matrix. To keep things time efficient,
# I initialize this matrix once and use it as necessary.
U = InitializeU()



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
    startingPoint = CoherentStateExact2D(x0, ky, 0)
    return startingPoint


# Implementation of coherent state specified in the
#
# https://files.slack.com/files-pri/T5MM8M0CR-F02LPDYN02X/download/2021-11-09-2d_coherent_state.pdf
#
# with initial x and y displacements
def CoherentStateExact2D(xinitial, ky, t):
    result = np.zeros((ny, nx), dtype=complex)
    for i in range(ny):
        for j in range(nx):
            currentx = x[j]
            currenty = y[i]

            normalization_const = (kappa**0.125)/(np.pi**0.5)

            x_part = np.exp( -0.5 * ( currentx-xinitial*np.cos(t) )**2  ) * np.exp( -1j * currentx * xinitial  * np.sin(t) )


            y_part = np.exp( (-0.5*np.sqrt(kappa)) * (currenty - (ky/np.sqrt(kappa))*np.sin(t * np.sqrt(kappa)) )**2  )       * np.exp( 1j * ky * currenty * np.cos(t*np.sqrt(kappa))   )

            result[i][j] = normalization_const * x_part * y_part

    return result



#1st checkpoint
psi1 = CoherentStateExact2D(0, 0, 0)
psi2 = CoherentStateExact2D(0, 0, 0)
print(Overlap2D(psi1, psi2))






def Run():
    terminateAt = number_of_oscillations * 2 * np.pi
    timesteps = math.ceil(terminateAt / dt)

    currentDate  = datetime.datetime.now()

    with open(filename, 'a') as f:
        f.write('# Simulation started at ' + str(currentDate) + '\n')
        f.write('#\n')
        f.write('# Simulation details: \n')

        f.write('# x-initial = ' + str(x0) + ' \n')
        f.write('# k_y = ' + str(ky) + ' \n')

        f.write('# Lx = ' + str(Lx) + ' \n')


        f.write('# Ly = ' + str(Ly) + ' \n')


        f.write('# dt = ' + str(dt) + ' s \n')
        f.write('# dx = ' + str(dx) + ' \n')
        f.write('# dy = ' + str(dy) + ' \n')
        f.write('# Spatial grid points : ' + str(nx) + ' x ' + str(ny) + ' \n')
        f.write('#\n')
        f.write('# time\terror\n')

    psi_num = CoherentStateNumerical2D()

    for i in range(timesteps):
        currentTime = i*dt

        psi_exact = CoherentStateExact2D(x0, ky, currentTime)
        error = np.abs(1 - Overlap2D(psi_num, psi_exact))

        current_time_as_string = "{:.3f}".format(currentTime)
        print('Current time: ' + current_time_as_string)
        error_as_string = "{:.4e}".format(error)

        with open(filename, 'a') as f:
            f.write(current_time_as_string+'\t'+error_as_string+'\n')


        '''
        plt.suptitle('Time: ' + current_time_as_string + '\n ky = '+str(ky)+' , x_initial = '+str(x0)+' \nError: ' + error_as_string)

        plt.subplot(1, 2, 1)
        plt.contourf(x, y, np.abs(psi_num)**2, 256, cmap='RdGy')
        plt.title('Numerical solution')
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.contourf(x, y, np.abs(psi_exact)**2, 256, cmap='RdGy')
        plt.title('Exact solution')
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.grid()

        plt.tight_layout()
        plt.savefig('visual'+str(i)+'.png')
        plt.clf()

        '''

        psi_num = Evolve(psi_num)


    end = time.time()
    difference = int(end - start)
    print( str(difference) + 's')
    with open(filename, 'a') as f:
        f.write('# Simulation finished at ' + str(currentDate) + '\n')
        f.write('# Simulation took ' + str(difference) + ' seconds\n')
