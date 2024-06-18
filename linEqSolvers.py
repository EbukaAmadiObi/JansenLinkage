import numpy as np

# Two functions for solving systems of linear equations: Gaussian Elimination and Gauss-Seidel Iteration.

# Written by Chukwuebuka Amadi-Obi for EEEN30150 Modelling and Simulation: Major Project 1

# Created 09/04/2024


# Define function
def gauss_elim(matrix_start, cons, verbose = False):
    """Performs Gaussian elimination (patrial pivoting) on a given system of linear equations in matrix form.

    Args:
        matrix: A numpy array representing the n*n matrix to be solved. (floats or complex numbers)
        cons: vector of constants
        verbose: A booelan, set to true if a detailed output of operations is desired. (defaults to false)

    Returns:
        - A numpy array containing the values for each variable in their respective index
    """

    #get number of variables
    N = len(matrix_start[0])

    #get number of equations
    M = len(matrix_start)

    matrix = np.zeros((M,N+1), dtype = cons.dtype)

    # add constant vector to end of matrix
    for i in range(M):
        matrix[i] = np.append(matrix_start[i],cons[i])

    if verbose:
        print(f"****************************************************\n\n\
                            Starting matrix:\n{matrix}\n\n\
                ****************************************************\n")

    # --------------------------------- Pivoting --------------------------------- #
    #for all variables
    for i in range(N):
        max_row=i
        #find maximum coefficient of current variable
        for j in range(i,M):
            if abs(matrix[j][i])>abs(matrix[max_row][i]):
                max_row=j

        #pivot by swapping rows
        matrix[[i,max_row]] = matrix[[max_row,i]]

        if verbose:
            print(f"swapped rows {i} and {max_row}: \n{matrix}\n")

        #subtract multiples of selected row from all others
        divisor = matrix[i][i]
        for k in range(i+1,M):
            fact = matrix[k][i]/divisor
            matrix[k]=matrix[k]-(matrix[i]*fact)

        if verbose:
            print(f"subtracted multiples of row {i} from all others: \n{matrix}\n\n\
                    ****************************************************\n")

    # ----------------------------- Back Substitution ---------------------------- #
    #for all variables (working backwards from the bottom)
    for i in range(N-1,-1,-1):
        if matrix[i][i]!=1:     #if not already, divide row by value to get variable value
            matrix[i]/=matrix[i][i]

        if verbose:
            print(f"Variable {i} is equal to {round(float(matrix[i][-1]),3)}")

        # From all rows above it, subtract multiple of current row to remove variable
        for j in range(i-1,-1,-1):

            matrix[j]-=matrix[i]*matrix[j][i]

        if verbose and i!=0:
            print(f"Back substituted variable {i} \n{matrix}\n")

    # Select value column and load into result value
    result = matrix[:,[-1]][:N].reshape((1,N))[:N].reshape((1,N))[0]
    if verbose:
        print(f"\n****************************************************\n\n\
                      Identity matrix found: \n{matrix}\n\n Variable values are: \n{result}\n")
    return result   #return result list

# Define function
def gauss_seidel(A, b, initial_guess, tolerance=1e-9, max_iterations=100, verbose=False):
    """Performs Gauss-Seidel Iteration on a given system of linear equations in matrix form.

    Args:
        A : Coefficient matrix.
        b : Constant vector.
        initial_guess : Initial guess for the solution vector.
        tolerance : Tolerance level for convergence. Default is 1e-6.
        max_iterations : Maximum number of iterations allowed. Default is 1000.

    Returns:
        - A numpy array containing the values for each variable in their respective index
    """

    if verbose:
        np.set_printoptions(precision=3)

    A_transpose = np.transpose(A)  # Transpose of A for preconditioning
    A = np.dot(A_transpose, A)
    b = np.dot(A_transpose, b)  # Preconditioned constant vector
    
    x = initial_guess.copy()   #make a copy of the initial guess to avoid modifying it
    x_new = np.zeros_like(x)
    n = len(x)    

    LU = np.tril(A,-1)+np.triu(A,1)

    for i in range(max_iterations):  #iterate until convergence or maximum iterations reached

        if verbose:
            print(f"\n**********ITERATION: {i+1}**********")

        for j in range(n):

            if verbose:
                print(f"x[{j}] = ({b[j]} + {-1*LU[j]}*{x_new}) divided by {A[j][j]}",end="")

            x_new[j] = (b[j]+np.sum(np.dot(x_new,-1*LU[j])))/A[j][j]
            
            if verbose:
                print(f" (Equals {x_new[j]})")

        if (all((x_new - x) < tolerance)):
            if verbose:
                print(f"Tolerance reached, estimated: \n {x_new}")
            return x_new
            
        x = x_new.copy()
    
    if (all((x_new - x) < tolerance)):
        if verbose:
            print(f"Iteration limit reached, estimated: \n {x_new}")
        return x_new
