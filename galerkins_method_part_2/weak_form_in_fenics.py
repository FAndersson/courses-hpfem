import matplotlib.pyplot as plt

from dolfin import UnitSquareMesh
from dolfin.common.plotting import plot
from dolfin.fem.bcs import DirichletBC
from dolfin.fem.solving import solve
from dolfin.functions import FunctionSpace, TrialFunction, TestFunction, Function, Expression, FacetNormal
from ufl import dx, ds, grad, dot, inner


def solve_equation():
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "Lagrange", 1)

    def boundary(x, on_boundary):
        return on_boundary

    u0 = Expression("cos(10 * x[0])", degree=2)
    bc = DirichletBC(V, u0, boundary)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("(x[0] - 0.5)*(x[0] - 0.5)", degree=2)

    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx

    u = Function(V)
    solve(a == L, u, bc)

    # Plot solution at current time step
    fig = plt.figure()
    plot(u, fig=fig)
    plt.show()

    print("The norm of u is {}".format(u.vector().norm("l2")))
