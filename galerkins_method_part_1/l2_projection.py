import matplotlib.pyplot as plt

from dolfin import UnitSquareMesh
from dolfin.common.plotting import plot
from dolfin.fem.assembling import assemble
from dolfin.fem.solving import solve
from dolfin.functions import FunctionSpace, TestFunction, Function, Expression
from ufl import dx


def simple_test():
    # Declare mesh and FEM functions
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    f = Expression("1.0 + sin(10 * x[0]) * cos(7 * x[1])", degree=3)
    v = TestFunction(V)
    u = Function(V)  # FEM solution

    # Weak form of L2 projection
    r = (u - f) * v * dx

    # Solving the linear system generated by the L2 projection
    solve(r == 0, u)

    # Plot the FEM solution
    fig = plt.figure()
    plot(u, fig=fig)
    plt.show()
    print(u.vector().norm('linf'))


def plot_basis_function():
    # Declare mesh and FEM functions
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    phi = Function(V)
    phi.vector()[:] = 0.0
    phi.vector()[10] = 1.0

    # Plot the basis function
    fig = plt.figure()
    plot(phi, fig=fig)
    plt.show()

    val = assemble(phi * phi * dx)
    print(val)
