import torch

# The code below implements the Newton-Raphson method to find a root of the function 
#   f(x) = x^3 - 2x - 5 
# using PyTorch for automatic differentiation.
# It iteratively updates the guess for the root until convergence.
# The code also prints the intermediate values of x and f(x) at each iteration. 

def f(x):
    return x**3 - 2*x - 5  # Example function: x^3 - 2x - 5

# Derivative of f(x), not directly used since we leverage autograd
def df(x):
    return 3*x**2 - 2  # Derivative: 3x^2 - 2

x = torch.tensor(1.0, requires_grad=True)  # Initial guess

max_iters = 20
tolerance = 1e-6

for i in range(max_iters):
    y = f(x)
    y.backward()  # Compute gradient

    with torch.no_grad():
        x_new = x - y / x.grad  # Newton's update
        if torch.abs(x_new - x) < tolerance:
            break
        print(f"Iteration {i+1}: x = {x.item()}, f(x) = {y.item()}")
        x.copy_(x_new)
        x.grad.zero_()  # Reset gradient for next iteration
print(f"Root found: x = {x.item()}, f(x) = {f(x).item()}")
