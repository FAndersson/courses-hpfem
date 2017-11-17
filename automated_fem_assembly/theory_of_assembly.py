from dolfin import UnitIntervalMesh
from dolfin.functions import FunctionSpace, TrialFunction, TestFunction
from dolfin.fem.assembling import assemble
from ufl import dx, grad, dot


def local_mass_matrix():
    # Define mesh
    mesh = UnitIntervalMesh(1)

    # Define function space
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define bilinear form
    a = (dot(grad(u), grad(v)) + u * v) * dx

    # Assemble matrix
    A = assemble(a)

    print(A.array())

    # Bilinear form for just the mass matrix part
    a2 = u * v * dx

    # Assemble mass matrix
    A2 = assemble(a2)
    print(6 * A2.array())


def global_stiffness_matrix():
    # Define mesh
    mesh = UnitIntervalMesh(10)

    # Define function space
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define bilinear form
    a = dot(grad(u), grad(v)) * dx

    # Assemble matrix
    A = assemble(a)

    h = 1 / 10
    print(h * A.array())
