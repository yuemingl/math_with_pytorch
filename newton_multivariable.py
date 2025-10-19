import torch

def f(x):
    f1 = x[0]**2 + x[1]**2 - 4  # Circle equation
    f2 = x[0] - x[1]**2       # Parabola equation
    return torch.stack([f1, f2])    

def df(x):
    df1_dx0 = 2*x[0]
    df1_dx1 = 2*x[1]
    df2_dx0 = 1
    df2_dx1 = -2*x[1]
    return torch.tensor([[df1_dx0, df1_dx1], [df2_dx0, df2_dx1]], dtype=torch.float32)

x = torch.tensor([1.0, 1.0], requires_grad=True)  # Initial guess
#x = torch.tensor([1.0, -1.0], requires_grad=True)  # Initial guess

max_iters = 20
tolerance = 1e-6

options = ['solve', 'inverse']
option = options[0]  # Choose 'solve' or 'inverse'
print("Using option:", option)

for i in range(max_iters):
    y = f(x)
    #J = df(x)  # Jacobian
    J = torch.autograd.functional.jacobian(f, x)  # Compute Jacobian using autograd

    dx = torch.linalg.solve(J, y)  # Solve J * dx = y

    print(f"Iteration {i+1}: x = {x.tolist()}, f(x) = {y.tolist()}")

    if option == 'solve':
        with torch.no_grad():
            x_new = x - dx
            if torch.norm(x_new - x) < tolerance:
                break
            x.copy_(x_new)
            # x.grad.zero_()  
            # No need to reset gradient since it is not like loss.backward(), 
            # it is not accumulating gradients
        print(f"Root found: x = {x.tolist()}, f(x) = {f(x).tolist()}")
    elif option == 'inverse':
        with torch.no_grad():
            # Alternative approach using inverse Jacobian (less efficient)
            J_inv = torch.linalg.inv(J)  # Inverse Jacobian
            print("J_inv:", J_inv)
            with torch.no_grad():
                x_new = x - (J_inv @ y.unsqueeze(1)).squeeze()
                if torch.norm(x_new - x) < tolerance:
                    break
                x.copy_(x_new)
        print(f"Root found: x = {x.tolist()}, f(x) = {f(x).tolist()}")

