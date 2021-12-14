#!/usr/bin/python
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import math
import datetime
import scipy.signal


# Simulation Parameters

dt = 1e-2   # Temporal seperation

# Spatial grid points
nx = 31
ny = 31

Lx = 10    # Spatial size, symmetric with respect to x=0
Ly = 10   # Spatial size, symmetric with respect to x=0

alpha=9

kx = 1
ky = alpha * kx

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
            sum += some_psi[i][j] * np.conj(some_psi[i][j]) * dx * dy

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
            psi = (alpha**(0.25))*(np.exp(-0.5*(currentx**2)))*((np.exp(-0.5*(alpha**0.5)*(currenty**2))))
            result[i][j] = psi

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
    return overlap

# This function calculates the overlap between two 2D wavefunctions
def Overlap1D(psi1, psi2):
    overlap = 0
    for i in range(x.size):
        overlap += psi1[i]*np.conj(psi2[i])*dx
    return overlap




def CoherentStateNumerical2D():
    return ExcitedStateNumerical2D(0) * np.exp(1j*kx*x) * np.exp(1j*ky*y)


# Implementation of coherent state specified in the
#
# https://files.slack.com/files-pri/T5MM8M0CR-F02LPDYN02X/download/2021-11-09-2d_coherent_state.pdf
#
# with x_0=0
def CoherentStateExact2D(t):
    result = np.zeros((ny, nx), dtype=complex)
    for i in range(ny):
        for j in range(nx):
            currentx = x[j]
            currenty = y[i]
            A = (alpha**0.25)/(np.pi**0.5)
            X = np.exp(  -0.5 * (currentx - 0*np.cos(t))**2    ) * np.exp(  -1j * 0 * currentx * np.sin(t))
            Y = np.exp(  -(0.5 * (alpha**0.5)) * (currenty - ky*np.sin(t*(alpha**0.5)))**2    ) * np.exp(  1j * ky * currenty * np.cos(t**(alpha**0.5)))
            result[i][j] = A*X*Y

    return result





'''
# x = 0 cross section

currentTime = 0
mid = ( ((nx+1)/2)+1 )
psi1 = Normalize2D(CoherentStateExact2D(currentTime))[:,11]
psi2 = Normalize2D(ExcitedStateNumerical2D(0))[:,11]
psi3 = Normalize2D(GroundStateExact2D())[:,11]


#print('Overlap between Exact coherent state(t=0) & Numerical ground state: ' + str(np.abs(Overlap1D(psi1,psi2))))
#print('Overlap between Exact coherent state(t=0) & Exact ground state: ' + str(np.abs(Overlap1D(psi1,psi3))))
#print('Numerical ground state: & Exact ground state: ' + str(np.abs(Overlap1D(psi2,psi3))))

plt.suptitle('x=0 cross section')


plt.plot(y, np.abs(psi1)**2, 'r-' , linewidth=6, label ='Exact coherent state at t=0')
plt.plot(y, np.abs(psi2)**2, 'b--' ,linewidth=4, label ='Numerical ground state')
plt.plot(y, np.abs(psi3)**2, 'y--' ,linewidth=2, label ='Exact ground state')
plt.grid()
plt.title('nx = ' + str(nx))

plt.xlabel('y')
plt.ylabel('Amplitude square')
plt.tight_layout()
plt.legend()

plt.show()
'''














'''
currentTime = 0
psi1 = Normalize2D(CoherentStateExact2D(currentTime))
psi2 = Normalize2D(ExcitedStateNumerical2D(0))

ov = np.abs(Overlap(psi1, psi2))

print('Timestep: ' + str(currentTime))

plt.suptitle('Overlap: {0:.7f}'.format(ov))

plt.subplot(1, 2, 1)
plt.contourf(x,y, np.abs(psi1)**2,256, cmap='RdYlBu')
plt.grid()
plt.title('Exact Ground State (Time:{0:.3f}s)'.format(currentTime))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.contourf(x,y, np.abs(psi2)**2,256, cmap='RdYlBu')
plt.grid()
plt.title('Numerical Ground State (Time:{0:.3f}s)'.format(currentTime))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()

plt.show()
'''











frames = 1200

num_psi = Normalize2D(CoherentStateNumerical2D())

for i in range(frames):
    currentTime = i*dt
    psi = CoherentStateExact2D(currentTime)
    print('Timestep: ' + str(currentTime))

    ov = np.abs(Overlap2D(psi, num_psi))

    plt.suptitle('Coherent State Exact vs. Numerical\nOverlap: {0:.7f}'.format(ov))

    plt.subplot(1, 2, 1)
    plt.contourf(x,y, np.abs(psi)**2,256, cmap='RdYlBu')
    plt.grid()
    plt.title('Exact solution (Time:{0:.3f}s)'.format(currentTime))
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.contourf(x,y, np.abs(num_psi)**2,256, cmap='RdYlBu')
    plt.grid()
    plt.title('Numerical solution (Time:{0:.3f}s)'.format(currentTime))
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    num_psi = Evolve(num_psi)

    plt.savefig('num_vs_exact' + str(i) +'.png')
    plt.clf()
