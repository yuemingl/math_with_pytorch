# PyTorch Newton's Method Implementations

This repository contains implementations of Newton's method using PyTorch for both single-variable and multi-variable optimization problems.

## Files

1. `newton.py` - Implementation of Newton's method for single-variable functions
2. `newton_multivariable.py` - Implementation of Newton's method for multi-variable systems of equations
3. `loss_backword.py` - Example using PyTorch's autograd for gradient computation

## Newton's Method for Multi-variable Systems

The `newton_multivariable.py` file implements Newton's method to find the intersection of:
- A circle: x₁² + x₂² = 4
- A parabola: x₁ = x₂²

The implementation provides two solution approaches:
1. Using `torch.linalg.solve` to solve the linear system J * dx = y
2. Using matrix inversion with `torch.linalg.inv` to compute x_new = x - J⁻¹y

Key features:
- Automatic differentiation with `torch.autograd.functional.jacobian`
- Custom implementation of the Jacobian matrix
- Convergence checking with tolerance
- Maximum iteration limit

## Requirements

- Python 3.x
- PyTorch

## Usage

To run the multi-variable Newton's method:
```bash
python newton_multivariable.py
```

The implementation will find the intersection points of the circle and parabola, demonstrating the convergence of Newton's method for systems of nonlinear equations.
