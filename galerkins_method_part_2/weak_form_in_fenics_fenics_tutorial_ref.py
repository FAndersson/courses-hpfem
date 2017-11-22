""" Code from the Fenics tutorial modified to solve the problem at hand"""

from fenics import *

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression("cos(10 * x[0])", degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("(x[0] - 0.5) * (x[0] - 0.5)", degree=2)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

print("The norm of u is {}".format(u.vector().norm("l2")))
