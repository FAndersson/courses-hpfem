import matplotlib.pyplot as plt

from dolfin import UnitSquareMesh
from dolfin.common.plotting import plot
from dolfin.fem.bcs import DirichletBC
from dolfin.fem.solving import solve
from dolfin.functions import FunctionSpace, TrialFunction, TestFunction, Function, Expression, FacetNormal
from ufl import dx, ds, grad, dot, inner


# TODO: Haven't been able to get the correct result for this one. Idea: Use solver for same problem in Fenics
# tutorial and compare with my solution


def solve_equation_1():
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "Lagrange", 1)

    def boundary(x, on_boundary):
        return on_boundary

    u0 = Expression("cos(10 * x[0])", degree=2)
    bc = DirichletBC(V, u0, boundary)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("(x[0] - 0.5)*(x[0] - 0.5)", degree=2)

    a = -inner(grad(u), grad(v)) * dx
    L = f * v * dx

    u = Function(V)
    solve(a == L, u, bc)

    # Plot solution at current time step
    fig = plt.figure()
    plot(u, fig=fig)
    plt.show()

    print("The norm of u is {}".format(u.vector().norm("l2")))


def solve_equation_2():
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "Lagrange", 1)

    def boundary(x, on_boundary):
        return on_boundary

    u0 = Expression("cos(10 * x[0])", degree=2)
    bc = DirichletBC(V, u0, boundary)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("(x[0] - 0.5)*(x[0] - 0.5)", degree=2)

    # Missing variational formulation
    eps = 1e-5
    x0b = Expression("x[0] < eps ? 10 * sin(10 * x[0]) : 0.0", eps=eps, degree=2)
    x1b = Expression("x[0] > 1.0 - eps ? -10 * sin(10 * x[0]) : 0.0", eps=eps, degree=2)

    a = -inner(grad(u), grad(v)) * dx + x0b * v * ds + x1b * v * ds
    L = f * v * dx

    u = Function(V)
    solve(a == L, u, bc)

    # Plot solution at current time step
    fig = plt.figure()
    plot(u, fig=fig)
    plt.show()

    print("The norm of u is {}".format(u.vector().norm("l2")))
