# secantMethodFindRoot.py
# finfo : 
# raise : 


import numpy as np

def secant_method(f, x0, x1, tol=1e-6, max_iter=50):
    """
    Find a root of the function f using the secant method.

    Parameters:
    f : function
        The function for which we are trying to find a root.
    x0 : float
        The first initial guess.
    x1 : float
        The second initial guess.
    tol : float
        The tolerance for convergence.
    max_iter : int
        The maximum number of iterations to perform.

    Returns:
    float
        The estimated root of the function f.
    """
    for iteration in range(max_iter):
        if abs(f(x1) - f(x0)) < np.finfo(float).eps: #if the difference of f(x1) - f(x0) is samller than machine epsilon, epsilon ==> enough small number to avoid division by zero in denominator.
            raise ValueError("Denominator in secant method is too small.") # by using raise we can raise(== throw an error) an exception when the condition is met.

        # Secant method formula
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

        # Check for convergence
        if abs(x2 - x1) < tol: #if the difference between current and previous estimate is less than tolerance/
            return x2 #return the estimated root

        # Update points for next iteration
        x0, x1 = x1, x2
        print("x2 == ", x2) #print the current estimate of the root at each iteration.
    raise ValueError("Secant method did not converge within the maximum number of iterations.")

def example_function(x): #example function is first derivative function
    return x**3 - 2*x - 5

x0,x1 = 2.0, 3.0

root = secant_method(example_function, x0, x1)
print(f"Estimated root: {root}")
print(f"Function value at root: {example_function(root)}")