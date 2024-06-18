import numpy as np
from linEqSolvers import gauss_elim


def curve_fit(x, y):
    """
    Curve fitting function that minimizes mean square error using Fermat's Theorem.

    Args:
        x (numpy array): Independent variable values.
        y (numpy array): Dependent variable values.

    Returns:
        - A list of 5th order polynomial coefficients. 

          In the form a + bx +cx^2 + dx^3, ex^4, fx^5, function 
          returns [a, b, c, d, e, f]
    """

    x=x.astype("float64")   # Convert data to large floats for easier manipulation
    y=y.astype("float64")

    lin_eq_mat = np.zeros([6,6])    #array for linear equation coefficient
    cons_mat = np.zeros([6,1])      #array for constants of linear equations

    for i in range(6):      # create 6 linear equations using Fermat's theorem
        k = sum(y*(x**i))
        k0 = sum(x**i)
        k1 = sum(x**(1+i))
        k2 = sum(x**(2+i))
        k3 = sum(x**(3+i))
        k4 = sum(x**(4+i))
        k5 = sum(x**(5+i))

        lin_eq_mat[i] = [k5,k4,k3,k2,k1,k0]     # Update values in coefficient and constant arrays
        cons_mat[i] = k

    return gauss_elim(lin_eq_mat,cons_mat)  # return answers in polynomial form

