#%% Main imports and functions
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import time

start_time = time.perf_counter()

CONST_V = .1  # standard potential used everywhere
CONST_W = 20  # standard width of the barrier unless specified otherwise
CONST_L = 50  # standard width of the surrounding infinite square well
CONST_N = 500  # standard number of dimensions for precision

# Define the potential for a rectangular barrier
def rectangular_potential(x, L, W, V):
    if -(L/2) < x < L/2:
        if -(W/2) < x < W/2:
            return V
        else:
            return 0
    else:
        return 100_000

# Define the potential for a triangular barrier
def triangular_potential(x,L,W,V):
    if -(L/2) < x < L/2:
        if -(W/2) < x < W/2:
            return V - V*2/W * abs(x)
        else:
            return 0
    else:
        return 100_000

# Define the potential for a Gaussian-shaped barrier
def gaussian_potential(x,L,W,V):
    if -(L/2) < x < L/2:
        if -(W/2) < x < W/2:
            return V*np.exp(np.log(.0001/V)*x**2*(2/W)**2)
        else:
            return 0
    else:
        return 100_000

# Define the potential for a reverse triangular barrier
def triangular_2_potential(x,L,W,V):
    if -(L/2) < x < L/2:
        if -(W/2) < x < W/2:
            return 2*V/W * abs(x)
        else:
            return 0
    else:
        return 100_000

# Define the potential roughly in the form of two Gaussian-shaped peaks
def double_gaussian_potential(x,L,W,V):
    if -(L/2) < x < L/2:
        if -(W/2) < x < W/2:
            return 1/2*V*(np.sin(x*np.pi*3/W-np.pi/2)+np.absolute(np.sin(x*np.pi*3/W-np.pi/2)))
        else:
            return 0
    else:
        return 100_000

# Define the potential for a linear combination of the triangular and the double Gaussian barriers
def linear_combination_potential(x,L,W,V):
    return 29/40*(triangular_potential(x,L,W,V) + double_gaussian_potential(x,L,W,V))

# Define a potential for a regular infinite square well (for testing purposes)
def infinite_square_well_potential(x,L,W,V):
    if -(L/2) < x < L/2:
        return 0
    else:
        return 10_000_000
    # return 0 if -L/2 < x < L/2 else 100_000

# Define a function to generate the matrix representation of H for a given function f
def generate_H_matrix(f,N,L,W,V):
    arr = np.zeros((N,N))  # temporary matrix to store results in
    x = np.linspace(-L/2, L/2, N)
    a = L/(N-1)  # value for a
    t = 1/(2*a**2)  # value for t
    # Fill in the matrix, see method for derivations of these values
    for i in range(N):
        if i == 0:
            arr[0][0]= 2*t + f(x[i],L,W,V)
            arr[0][1]=-t
        elif i == N-1:
            arr[N-1,N-1] = 2*t + f(x[i],L,W,V)
            arr[N-1, N-2] = -t
        else: 
            arr[i,i] = 2*t + f(x[i],L,W,V)
            arr[i,i-1] = -t
            arr[i,i+1] = -t
    return arr, x  # note that x is returned simply to have a value for the linspace which we use often for convenience


# Generate the matrices for every potential (except the ISW which we do later)
temp1 = generate_H_matrix(rectangular_potential,CONST_N,CONST_L,CONST_W,CONST_V)
temp2 = generate_H_matrix(triangular_potential,CONST_N,CONST_L,CONST_W,CONST_V)
temp3 = generate_H_matrix(gaussian_potential,CONST_N,CONST_L,CONST_W,CONST_V)
temp4 = generate_H_matrix(triangular_2_potential,CONST_N,CONST_L,CONST_W,CONST_V)
temp4 = generate_H_matrix(triangular_2_potential,CONST_N,CONST_L,CONST_W,CONST_V)
temp5 = generate_H_matrix(double_gaussian_potential,CONST_N,CONST_L,CONST_W,CONST_V)
temp6 = generate_H_matrix(linear_combination_potential,CONST_N,CONST_L,CONST_W,CONST_V)

# Calculate the eigenenergies and eigenfunctions of each matrix
E1,psi1 = np.linalg.eigh(temp1[0])
E2,psi2 = np.linalg.eigh(temp2[0])
E3,psi3 = np.linalg.eigh(temp3[0])
E4,psi4 = np.linalg.eigh(temp4[0])
E5,psi5 = np.linalg.eigh(temp5[0])
E6,psi6 = np.linalg.eigh(temp6[0])

# Define a function to integrate the square of the n-th eigenfunction of a wave function inside the barrier
def integrate(f,N,L,W,V,n, eigenfunctions):
    H_matrix = generate_H_matrix(f, N, L, W, V)  # generate the H-matrix
    func = eigenfunctions[:,n]**2  # take the n-th vector and square it
    mask_1 = H_matrix[1] < W/2  # look only at the values to the left of the barrier
    mask_2 = H_matrix[1] > -(W/2)  # look only at the values to the right of the barrier
    func = func[mask_1 & mask_2]  # apply the masks
    return scipy.integrate.cumulative_trapezoid(func, initial=0)[-1]  # integrate the resulting vector


#%% The eigenenergy as a function of the number of the eigenstate
fig,axs = plt.subplots(2,3, sharex=False, sharey= True)

axs[0,0].plot(E1[E1<50000], 'ko', markersize=0.5)
axs[0,1].plot(E2[E2<50000], 'ko', markersize=0.5)
axs[0,2].plot(E3[E3<50000], 'ko', markersize=0.5)
axs[1,0].plot(E4[E4<50000], 'ko', markersize=0.5)
axs[1,1].plot(E5[E5<50000], 'ko', markersize=0.5)
axs[1,2].plot(E6[E6<50000], 'ko', markersize=0.5)

axs[1,0].set_xlabel(r'$Eigenstate$')
axs[1,1].set_xlabel(r'$Eigenstate$')
axs[1,2].set_xlabel(r'$Eigenstate$')

axs[0,0].set_ylabel(r'$Eigenenergy$ (E$_h$)')
axs[1,0].set_ylabel(r'$Eigenenergy$ (E$_h$)')

fig.suptitle('Eigenenergy as a function of number of eigenstate')
plt.savefig('Eigenenergy as a function of number of eigenstate.pdf',
            dpi=3000, format='pdf',bbox_inches='tight'  )
#%% The potential as a function of position

# Define a function to variate the x-values for a certain function
def variate_x(function, L=CONST_L, W=CONST_W, V=CONST_V, N=CONST_N):
    x = np.linspace(-(L/2), L/2, N)
    return [function(i,L,W,V) for i in x]

# Calculate all the potentials at every x
y_rectangular_potential= variate_x(rectangular_potential)
y_rectangular_potential[0] = max(y_rectangular_potential[1:-1]) #for illustrative purposes
y_rectangular_potential[-1] = max(y_rectangular_potential[1:-1])
y_triangular_potential = variate_x(triangular_potential)
y_triangular_potential[0] = max(y_triangular_potential[1:-1]) #for illustrative purposes
y_triangular_potential[-1] = max(y_triangular_potential[1:-1])
y_smooth_potential = variate_x(gaussian_potential)
y_smooth_potential[0] = max(y_smooth_potential[1:-1]) #for illustrative purposes
y_smooth_potential[-1] = max(y_smooth_potential[1:-1])
y_triangular_2_potential= variate_x(triangular_2_potential)
y_triangular_2_potential[0] = max(y_triangular_2_potential[1:-1]) #for illustrative purposes
y_triangular_2_potential[-1] = max(y_triangular_2_potential[1:-1])
y_double_gaussian_potential= variate_x(double_gaussian_potential)
y_double_gaussian_potential[0] = max(y_double_gaussian_potential[1:-1]) #for illustrative purposes
y_double_gaussian_potential[-1] = max(y_double_gaussian_potential[1:-1])
y_linear_combination_potential= variate_x(linear_combination_potential)
y_linear_combination_potential[0] = max(y_linear_combination_potential[1:-1]) #for illustrative purposes
y_linear_combination_potential[-1] = max(y_linear_combination_potential[1:-1])

# Plot the results
fig,axs = plt.subplots(2,3, sharex=True, sharey= True)
axs[0,0].plot(temp1[1], y_rectangular_potential)
axs[0,1].plot(temp1[1], y_triangular_potential)
axs[0,2].plot(temp1[1], y_smooth_potential)
axs[1,0].plot(temp1[1], y_triangular_2_potential)
axs[1,1].plot(temp1[1], y_double_gaussian_potential)
axs[1,2].plot(temp1[1], y_linear_combination_potential)
axs[1,0].set_xlabel(r'$Position$ (x)')
axs[1,1].set_xlabel(r'$Position$ (x)')
axs[1,2].set_xlabel(r'$Position$ (x)')
axs[0,0].set_ylabel(r'$Potential$ (E$_h$)')
axs[1,0].set_ylabel(r'$Potential$ (E$_h$)')

fig.suptitle('Potential as a function of position')
plt.savefig('Potential as a function of position.pdf',
            dpi=3000, format='pdf',bbox_inches='tight')

#%% Probability of a particle being in barrier as a function of energy

# Define a function to integrate a function along a given amount of eigenstates
def integrate_states(function, states, eigenfunctions, N=CONST_N, L=CONST_L, W=CONST_W, V=CONST_V):
    return np.array([integrate(function, N, L, W, V, state, eigenfunctions) for state in states])

number_of_states = CONST_N
states = np.arange(number_of_states)  # all the numbers of the eigenstates
# Integrate every potential along every eigenstate. Note that the masks are to make sure the states are bound
area1 = integrate_states(rectangular_potential, states, psi1)[E1[:number_of_states]<50000]
area2 = integrate_states(triangular_potential, states, psi2)[E2[:number_of_states]<50000]
area3 = integrate_states(gaussian_potential, states, psi3)[E3[:number_of_states]<50000]
area4 = integrate_states(triangular_2_potential, states, psi4)[E4[:number_of_states]<50000]
area5 = integrate_states(double_gaussian_potential, states, psi5)[E5[:number_of_states]<50000]
area6 = integrate_states(linear_combination_potential, states, psi6)[E6[:number_of_states]<50000]

# Plot the results
fig,axs = plt.subplots(2,3, sharex=False, sharey= True)
axs[0,0].plot(E1[E1<50000][:number_of_states], area1, 'ko', markersize=1)
axs[0,1].plot(E2[E2<50000][:number_of_states], area2, 'ko', markersize=1)
axs[0,2].plot(E3[E3<50000][:number_of_states], area3, 'ko', markersize=1)
axs[1,0].plot(E4[E4<50000][:number_of_states], area4, 'ko', markersize=1)
axs[1,1].plot(E5[E5<50000][:number_of_states], area5, 'ko', markersize=1)
axs[1,2].plot(E6[E6<50000][:number_of_states], area6, 'ko', markersize=1)

axs[1,0].set_xlabel(r'$Eigenenergy$ (E$_h$)')
axs[1,1].set_xlabel(r'$Eigenenergy$ (E$_h$)')
axs[1,2].set_xlabel(r'$Eigenenergy$ (E$_h$)')
axs[0,0].set_ylabel(r'$Probability$')
axs[1,0].set_ylabel(r'$Probability$')

fig.suptitle('Probability that the particle is within |x|<W/2 as a function of eigenenergy')
plt.savefig('Probability that the particle is within as a function of eigenenergy.pdf',
            dpi=3000, format='pdf',bbox_inches='tight')

#%% Probability of a particle being in barrier as a function of barrier width

# Define a function to variate the width of the barrier and return the integral along a given eigenstate n
def variate_width(function, widths, eigenfunctions, n, L=CONST_L, V=CONST_V, N=CONST_N):
    return np.array([integrate(function,N,L,width,V,n,eigenfunctions) for width in widths])

# Define a function to calculate the probability to find the particle in the barrier as a function of width, for a specific eigenstate n
def probability_from_width(n):
    widths = np.linspace(1,50)  # the widths along which to variate
    # Calculate the area for each potential and for each width for a given n
    area1_W = variate_width(rectangular_potential, widths, psi1, n)
    area2_W = variate_width(triangular_potential, widths, psi2, n)
    area3_W = variate_width(gaussian_potential, widths, psi3, n)
    area4_W = variate_width(triangular_2_potential, widths, psi4, n)
    area5_W = variate_width(double_gaussian_potential, widths, psi5, n)
    area6_W = variate_width(linear_combination_potential, widths, psi6, n)
    
    # Plot the results
    fig,axs = plt.subplots(2,3, sharex=True, sharey= True)
    axs[0,0].plot(widths, area1_W, 'ko',markersize=1)
    axs[0,1].plot(widths, area2_W,'ko',markersize=1)
    axs[0,2].plot(widths,  area3_W,'ko',markersize=1)
    axs[1,0].plot(widths,  area4_W,'ko',markersize=1)
    axs[1,1].plot(widths,  area5_W,'ko',markersize=1)
    axs[1,2].plot(widths, area6_W,'ko',markersize=1)
    
    axs[1,0].set_xlabel(r'$Barrier$ $width$ ')
    axs[1,1].set_xlabel(r'$Barrier$ $width$ ')
    axs[1,2].set_xlabel(r'$Barrier$ $width$')
    axs[0,0].set_ylabel(r'$Probability$')
    axs[1,0].set_ylabel(r'$Probability$')
    
    # Give the figure an appropriate name, based on which n we are calculating the probability for
    if  n==0:
      fig.suptitle(f'Probability that particle is within |x|<W/2 versus barrier width {n+1}st eigenstate ')
      plt.savefig(f'Probability that particle is within  versus barier width {n+1}st eigenstate.pdf',
            dpi=3000, format='pdf',bbox_inches='tight')
    elif n==1:
      fig.suptitle(f'Probability that particle is within |x|<W/2 versus barrier width {n+1}nd eigenstate')
      plt.savefig(f'Probability that particle is within  versus barrier width {n+1}nd eigenstate.pdf',
            dpi=3000, format='pdf',bbox_inches='tight')
    elif n==2:
     fig.suptitle(f'Probability that particle is within |x|<W/2 versus barrier width  {3}rd eigenstate')
     plt.savefig(f'Probability that particle is within  versus barrier width. {3}rd eigenstate.pdf',
            dpi=3000, format='pdf',bbox_inches='tight')
    else: 
     fig.suptitle(f'Probability that particle is within |x|<W/2 versus barrier width {n+1}th eigenstate ')
     plt.savefig(f'Probability that particle is within versus barrier width {n+1}th eigenstate.pdf',
            dpi=3000, format='pdf',bbox_inches='tight')
     
    fig.tight_layout()


# Some example calculations
probability_from_width(0)
probability_from_width(99)

#%% Potentials and probability densities in the same figures
# Define a function to plot the probability density and potential in the same figure for all potentials, and for a given eigenstate n
def subplotter(n):
    fig,axs = plt.subplots(2,3, sharex=True, sharey= True)
    axs[0,0].plot(temp1[1], y_rectangular_potential,'k-')
    ax00 = axs[0,0].twinx()
    ax00.plot(temp1[1],(psi1[:,n]**2),label='Prob density', markersize=0.001)
    
    axs[0,1].plot(temp1[1], y_triangular_potential,'k-')
    ax01 = axs[0,1].twinx()
    ax01.plot(temp1[1],(psi2[:,n]**2),label='Prob density', markersize=0.001)
    
    axs[0,2].plot(temp1[1], y_smooth_potential,'k-')
    ax02 = axs[0,2].twinx()
    ax02.plot(temp1[1],(psi3[:,n]**2),label='Prob density', markersize=0.001)
    
    axs[1,0].plot(temp1[1], y_triangular_2_potential,'k-')
    ax10 = axs[1,0].twinx()
    ax10.plot(temp1[1],(psi4[:,n]**2),label='Prob density', markersize=0.001)
    
    axs[1,1].plot(temp1[1], y_double_gaussian_potential,'k-')
    ax11 = axs[1,1].twinx()
    ax11.plot(temp1[1],(psi5[:,n]**2),label='Prob density', markersize=0.001)
    
    axs[1,2].plot(temp1[1], y_linear_combination_potential,'k-')
    ax12 = axs[1,2].twinx()
    ax12.plot(temp1[1],(psi6[:,n]**2),label='Prob density', markersize=0.001)
    
    axs[1,0].set_xlabel(r'$Position$(x)')
    axs[1,1].set_xlabel(r'$Position$(x)')
    axs[1,2].set_xlabel(r'$Position$(x)')
    axs[0,0].set_ylabel(r'$Potential$ (E$_h$)')
    axs[1,0].set_ylabel(r'$Potential$ (E$_h$)')
    
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Once again, give the figures appropriate names based on n
    if  n==0:
      fig.suptitle(f'Probability density function {n+1}st eigenstate (blue) and potential (black)')
      plt.savefig(f'Probability density function {n+1}st eigenstate (blue) and potential (black).pdf',
                  dpi=3000, format='pdf',bbox_inches='tight')
    elif n==1:
      fig.suptitle(f'Probability density function {n+1}nd eigenstate (blue) and potential (black)')
      plt.savefig(f'Probability density function {n+1}nd eigenstate (blue) and potential (black).pdf',
                  dpi=3000, format='pdf',bbox_inches='tight')
    elif n==2:
     fig.suptitle(f'Probability density function {3}rd eigenstate (blue) and potential (black)')
     plt.savefig(f'Probability density function {3}rd eigenstate (blue) and potential (black).pdf',
                  dpi=3000, format='pdf',bbox_inches='tight')
    else: 
     fig.suptitle(f'Probability density function {n+1}th eigenstate (blue) and potential (black)')
     plt.savefig(f'Probability density function {n+1}th eigenstate (blue) and potential (black).pdf',
                  dpi=3000, format='pdf',bbox_inches='tight')
     
# Some example plots
subplotter(19)
subplotter(9)
subplotter(0)


#%% Testing most things against the ISW (obviously cannot test for varying width of a barrier as there is none, also cannot test integration of the function inside the barrier as, once again, there is none)
ISW_matrix, ISW_linspace = generate_H_matrix(infinite_square_well_potential, CONST_N, CONST_L, CONST_W, CONST_V)  # generate the matrix for the ISW
E_ISW, psi_ISW = np.linalg.eigh(ISW_matrix)  # get the eigenenergy and eigenvectors

# Plot the first 4 eigenfunctions of the ISW
f1 = plt.figure()
for i in range(4):
    plt.plot(ISW_linspace, psi_ISW[:,i], label=f"Eigenfunction {i+1}")
plt.xlabel(r"$Position$ (x)")
plt.ylabel(r"$Value$ $of$ $the$ $wave$ $function$ ($\phi$(x))")
plt.legend(bbox_to_anchor=(.6,.3))
f1.suptitle("The first four eigenfunctions of the ISW")
plt.savefig("First four eigenfunctions of the ISW.pdf", dpi=3000, format="pdf", bbox_inches="tight")

# Do the same for the probability densities
f2 = plt.figure()
for i in range(4):
    plt.plot(ISW_linspace, psi_ISW[:,i]**2, label=f"Eigenfunction {i+1}")
plt.xlabel(r"$Position$ (x)")
plt.ylabel(r"$Probability$ $density$ (|$\phi$(x)|$^2$)")
plt.legend(loc="upper right")
f2.suptitle("Probability density of the first four eigenfunctions")
plt.savefig("Probability density first four eigenfunctions ISW.pdf", dpi=3000, format="pdf", bbox_inches="tight")

# Plot the eigenenergy as a function of number of eigenstate
f3 = plt.figure()
plt.plot(E_ISW[E_ISW<50_000], 'ko', markersize=.5)
plt.xlabel(r'$Eigenstate$')
plt.ylabel(r'$Eigenenergy$ (E$_h$)')
f3.suptitle("Eigenenergy as a function of number of eigenstate of ISW")
plt.savefig("Eigenenergy as a function of number of eigenstate of ISW.pdf", dpi=3000, format="pdf", bbox_inches="tight")


#%% How long the program took to execute
print(f"Elapsed time: {time.perf_counter() - start_time}")