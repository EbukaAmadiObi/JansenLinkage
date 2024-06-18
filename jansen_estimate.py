import numpy as np
from linEqSolvers import gauss_elim, gauss_seidel
from curve_fit import curve_fit
from numpy import cos, sin
import matplotlib.pyplot as plt
import matplotlib.animation as anim


# Python script to simulate motion of Theo Jansen's mechanical linkage

# Written by Chukwuebuka Amadi-Obi for EEEN30150 Modelling and Simulation: Major Project 1

# Created 11/04/2024

# ----------------------------- HELPER FUNCTIONS ----------------------------- #
def f(theta2, x):
    """
    Calculate value of f(x) for give values of theta2-theta12
    
    Args:
        theta2: Input angle of link 2 (in radians)
        x: length 10 array of theta variables
    
    Returns:
        y: value for all equations given variables
    """
    y = np.array([38+15*cos(theta2)+50*cos(x[0])+41.5*cos(x[1]),
           7.8+15*sin(theta2)+50*sin(x[0])+41.5*sin(x[1]),
           -41.5*cos(x[1])+55.8*cos(x[2])+40.1*cos(x[3]),
           -41.5*sin(x[1])+55.8*sin(x[2])+40.1*sin(x[3]),
           40.1*cos(x[3])-39.4*cos(x[4])+39.3*cos(x[5])-36.7*cos(x[6]),
           40.1*sin(x[3])-39.4*sin(x[4])+39.3*sin(x[5])-36.7*sin(x[6]),
           38+15*cos(theta2)-39.3*cos(x[5])-61.9*cos(x[7]),
           7.8+15*sin(theta2)-39.3*sin(x[5])-61.9*sin(x[7]),
           36.7*cos(x[6])+49*cos(x[8])+65.7*cos(x[9]),
           36.7*sin(x[6])+49*sin(x[8])+65.7*sin(x[9])])
    return y

def jf(x):
    """
    Calculate value of the jacobian of f(x) for given values of theta3-theta12
    
    Args:
        x: length 10 array of theta variables
    
    Returns:
        y: value for all equations given variables
    """
    y = np.array([[-50*sin(x[0]),-41.5*sin(x[1]),0,0,0,0,0,0,0,0],
            [50*cos(x[0]),41.5*cos(x[1]),0,0,0,0,0,0,0,0],
            [0,41.5*sin(x[1]),-55.8*sin(x[2]),-40.1*sin(x[3]),0,0,0,0,0,0],
            [0,-41.5*cos(x[1]),55.8*cos(x[2]),40.1*cos(x[3]),0,0,0,0,0,0],
            [0,0,0,-40.1*sin(x[3]),39.4*sin(x[4]),-39.3*sin(x[5]),36.7*sin(x[6]),0,0,0],
            [0,0,0,40.1*cos(x[3]),-39.4*cos(x[4]),39.3*cos(x[5]),-36.7*cos(x[6]),0,0,0],
            [0,0,0,0,0,39.3*sin(x[5]),0,61.9*sin(x[7]),0,0],
            [0,0,0,0,0,-39.3*cos(x[5]),0,-61.9*cos(x[7]),0,0],
            [0,0,0,0,0,0,-36.7*sin(x[6]),0,-49*sin(x[8]),-67.5*sin(x[9])],
            [0,0,0,0,0,0,36.7*cos(x[6]),0,49*cos(x[8]),67.5*cos(x[9])]])
    return y

# ----------------------- NEWTON RAPHSON IMPLEMENTATION ---------------------- #
def jansen_estimate(theta2, xn, solver=0, tolerance = 1e-9, max_iterations = 20, verbose = False):
    """Estimate roots of multivariate functions f(x)

    Args:
        theta2: Input angle of link 2 (in radians)
        xn: Initial guesses for angles of linkages 3-12 (in radians)
        solver: solve linear equations using Gaussian Elimination or Gauss-Siedel methods [int 0 or 1 respectively]
        tolerance: Max acceptable value of any f(x) values before iteration ends
        verbose: If true, print debug text to terminal
        max_iterations: Maximum number of iterations before force ending loop

    Returns:
        - A vector of variables at which f(x) has roots
    """

    if verbose: np.set_printoptions(precision=3); print(f"**********INITIAL GUESS**********\n{xn}\n")
    iteration_count = 0 # int for counting iterations
    while iteration_count < max_iterations: # loop until all iterations are finished
        if verbose: print(f"**********ITERATION {iteration_count+1}**********")

        fxn = f(theta2, xn) #calculate residual for guess
        jfxn = jf(xn)   #calculate jacobian for guess

        if np.linalg.norm(fxn) < tolerance:
            if verbose: print(f"Approximate solution is: {xn}\n")
            return xn
        
        if verbose: print(f"Residual is: {fxn}\n")

        if solver == 0:   #if using Gaussian Elimination
            deltax = gauss_elim(jfxn, -1*fxn)

        if solver == 1:
            deltax = gauss_seidel(jfxn, -1*fxn, xn, tolerance)

        xn+=deltax

        iteration_count+=1  #incease iteration count

# ----------------------- ANGLE ITERATION AND ANIMATION ---------------------- #
inital_guesses = np.array([2.618,  #initial estimates for angles of all vectors
                    -1.745,
                    -2.548,
                    -0.244,
                    -1.030,
                    -1.134,
                    -0.332,
                    0.977,
                    -1.745,
                    1.99])

no_points = 100
crank_angles = np.linspace(0,2*np.pi,no_points)   #range of 100 crank angles between 0 and 2pi rads
link2 ,link3 ,link4 ,link5 ,link6 ,link7 ,link8 ,link9 ,link10 ,link11 ,link12 = [],[],[],[],[],[],[],[],[],[],[]   #lists to hold representations of each linkage

trace = ([],[]) #list to hold x and y co-ordinates of each point on trace

for iter in range(len(crank_angles)):     #for all crank angles
    theta2 = crank_angles[iter]
    thetas= jansen_estimate(theta2,inital_guesses,0)     #estimate other linkage angles using Newton Raphson

    zero=[0,0]             # create two lists for x and y co-ords of each point using angle estimates
    one=[-38,-7.8]
    i=[15*cos(theta2),15*sin(theta2)]
    j=[-38-41.5*cos(thetas[1]),-7.8-41.5*sin(thetas[1])]
    k=[-38-40.1*cos(thetas[3]),-7.8-40.1*sin(thetas[3])]
    l=[-38-40.1*cos(thetas[3])+39.4*cos(thetas[4]),-7.8-40.1*sin(thetas[3])+39.4*sin(thetas[4])]
    m=[-38+39.3*cos(thetas[5]),-7.8+39.3*sin(thetas[5])]
    n=[-38+39.3*cos(thetas[5])+49*cos(thetas[8]),-7.8+39.3*sin(thetas[5])+49*sin(thetas[8])]

    link2.append([[zero[0],i[0]],[0,0],[zero[1],i[1]]])     # for each linkage add proper co-ords to list
    link3.append([[i[0],j[0]],[0,0],[i[1],j[1]]])
    link4.append([[j[0],one[0]],[0,0],[j[1],one[1]]])
    link5.append([[j[0],k[0]],[0,0],[j[1],k[1]]])
    link6.append([[k[0],one[0]],[0,0],[k[1],one[1]]])
    link7.append([[k[0],l[0]],[0,0],[k[1],l[1]]])
    link8.append([[one[0],m[0]],[0,0],[one[1],m[1]]])
    link9.append([[l[0],m[0]],[0,0],[l[1],m[1]]])
    link10.append([[i[0],m[0]],[0,0],[i[1],m[1]]])
    link11.append([[m[0],n[0]],[0,0],[m[1],n[1]]])
    link12.append([[n[0],l[0]],[0,0],[n[1],l[1]]])

    trace[0].append(n[0])       # append x and y co-ords of pointon trace for current crank angle
    trace[1].append(n[1])
    
fig = plt.figure()  #set up figure
ax = fig.add_subplot(projection='3d')  #Set up 3d projecction plot


ax.set_xlim(-140,50)        #set x and z limits
ax.set_zlim(-100,50)
#ax.set_axis_off()   #remove axis lines
lw = 4      #set line width
lc = "grey" #set line colour

def update(frame):      # update function to be called repeatedly
    for art in list(ax.lines):      #remove all art
        art.remove()
    ax.plot3D(*link2[frame],color = lc,linewidth=lw)    #draw all linkages
    ax.plot3D(*link3[frame],color = lc,linewidth=lw)
    ax.plot3D(*link4[frame],color = lc,linewidth=lw)
    ax.plot3D(*link5[frame],color = lc,linewidth=lw)
    ax.plot3D(*link6[frame],color = lc,linewidth=lw)
    ax.plot3D(*link7[frame],color = lc,linewidth=lw)
    ax.plot3D(*link8[frame],color = lc,linewidth=lw)
    ax.plot3D(*link9[frame],color = lc,linewidth=lw)
    ax.plot3D(*link10[frame],color = lc,linewidth=lw)
    ax.plot3D(*link11[frame],color = lc,linewidth=lw)
    ax.plot3D(*link12[frame],color = lc,linewidth=lw)
    ax.plot3D(trace[0],np.zeros(len(trace[0])),trace[1], color = "#b5b5b5",zorder=1)   #draw trace of bottom joint

ani = anim.FuncAnimation(fig, update, frames = len(link2), interval = 30)   # animation function, input figure to draw on, funciton to call, length of animaiton and time interval between frames
plt.show()  #show animation

# ------------------ CURVE FITTING OF UPPER AND LOWER TRACE ------------------ #

u_trace_x = np.array(trace[0][32:72])   # Split trace into upper and lower components
u_trace_y = np.array(trace[1][32:72])

l_trace_x = np.append(trace[0][71:100],trace[0][0:33])
l_trace_y = np.append(trace[1][71:100],trace[1][0:33])

x_range = np.linspace(l_trace_x[0],l_trace_x[-1],200)   # get range of x values to plot curves over
u_fit_curve = np.polyval(curve_fit(u_trace_x,u_trace_y),x_range)    # fit curves to upper and lower trace
l_fit_curve = np.polyval(curve_fit(l_trace_x,l_trace_y),x_range)

plt.plot(x_range,u_fit_curve)       #plot curves and original trace
plt.plot(x_range,l_fit_curve)
plt.scatter(u_trace_x,u_trace_y, color = "blue",s=4, alpha=0.1875)
plt.scatter(l_trace_x,l_trace_y, color = "red",s=4, alpha=0.1875)

plt.show()  #display fit curves with traces