import numpy as np

from dolfin import UnitSquareMesh, cells, Vertex
from dolfin.fem.assembling import assemble_system
from dolfin.functions import Constant, FunctionSpace, TrialFunction, TestFunction
from ufl import inner, grad, dx


def myassemble(mesh):
    # Define basis functions and their gradients on the reference elements.
    def phi0(x):
        return 1.0 - x[0] - x[1]

    def phi1(x):
        return x[0]

    def phi2(x):
        return x[1]

    def f(x):
        return 1.0

    phi = [phi0, phi1, phi2]
    dphi = np.array(([-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]))

    # Define quadrature points
    midpoints = np.array(([0.5, 0.0], [0.5, 0.5], [0.0, 0.5]))

    N = mesh.num_vertices()

    A = np.zeros((N, N))

    b = np.zeros((N, 1))

    # Used to hold cell vertices
    coord = np.zeros([3, 2])

    # Iterate over all cells, adding integral contribution to matrix/vector
    for c in cells(mesh):

        # Extract node numbers and vertex coordinates
        nodes = c.entities(0).astype('int')
        for i in range(0, 3):
            v = Vertex(mesh, int(nodes[i]))
            for j in range(0, 2):
                coord[i][j] = v.point()[j]

        # Compute Jacobian of map and area of cell
        J = np.outer(coord[0, :], dphi[0]) + \
            np.outer(coord[1, :], dphi[1]) + \
            np.outer(coord[2, :], dphi[2])
        dx = 0.5 * abs(np.linalg.det(J))

        # Iterate over quadrature points
        for p in midpoints:
            # Map p to physical cell
            x = coord[0, :] * phi[0](p) + \
                coord[1, :] * phi[1](p) + \
                coord[2, :] * phi[2](p)

            # Iterate over test functions
            for i in range(0, 3):
                v = phi[i](p)
                dv = np.linalg.solve(J.transpose(), dphi[i])

                # Assemble vector (linear form)
                integral = f(x)*v*dx / 3.0
                b[nodes[i]] += integral

                # Iterate over trial functions
                for j in range(0, 3):
                    u = phi[j](p)
                    du = np.linalg.solve(J.transpose(), dphi[j])

                    # Assemble matrix (bilinear form)
                    integral = (np.inner(du, dv)) * dx / 3.0
                    integral += u * v * dx / 3.0
                    A[nodes[i], nodes[j]] += integral

    return A, b


def assemble_example():
    mesh = UnitSquareMesh(5, 5)
    mesh.init()

    A, b = myassemble(mesh)

    print("The norm of A is {}".format(np.linalg.norm(A)))


def assemble_matrices_own_functions():
    import scipy as sp

    from finite_element.laplace_equation import assemble
    from finite_element.projection import assemble_lhs
    from geometry.mesh.basic_triangulations import rectangle_triangulation, rectangle_vertices

    n = 6
    triangles = rectangle_triangulation(n, n)
    vertices = rectangle_vertices(1, 1, n, n)[:, 0:2]
    r = 1
    A1 = assemble(triangles, vertices, r)
    A2 = assemble_lhs(triangles, vertices, r)
    A = A1 + A2

    print("The norm of A is {}".format(sp.sparse.linalg.norm(A)))


if __name__ == "__main__":
    # assemble_example()
    assemble_matrices_own_functions()
