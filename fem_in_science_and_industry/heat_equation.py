import matplotlib.pyplot as plt

from dolfin import Point
from dolfin.common.plotting import plot
from dolfin.fem import DirichletBC
from dolfin.fem.norms import errornorm
from dolfin.fem.projection import project
from dolfin.fem.solving import solve
from dolfin.functions import FunctionSpace, TestFunction, Function, Expression
from mshr import Rectangle, generate_mesh
from ufl import dx, inner, grad


def setup_geometry():
    # Generate mesh
    xmin, xmax = -2.0, 2.0
    ymin, ymax = -2.0, 2.0
    geometry = Rectangle(Point(xmin, ymin), Point(xmax, ymax))
    mesh_resolution = 20
    mesh = generate_mesh(geometry, mesh_resolution)

    # Define boundary
    def boundary(x, on_boundary):
        return on_boundary
        
    return mesh, boundary
    
    
def solve_heat_equation(k):
    """
    Solve the heat equation on a hard-coded mesh with a hard-coded initial and boundary conditions
    
    :param k: Thermal conductivity
    """
    mesh, boundary = setup_geometry()
    
     # Exact solution (Gauss curve)
    ue = Expression("exp(-(x[0]*x[0]+x[1]*x[1])/(4*a*t))/(4*pi*a*t)", a=k, t=1e-7, domain=mesh, degree=2)
    
    # Polynomial degree
    r = 1
    
    # Setup FEM function space
    V = FunctionSpace(mesh, "CG", r)
    
    # Create boundary condition
    bc = DirichletBC(V, ue, boundary)
    
    # Setup FEM functions
    v = TestFunction(V)
    u = Function(V)

    # Time parameters
    time_step = 0.5
    t_start, t_end = 0.0, 20.0

    # Time stepping
    t = t_start
    u0 = ue
    step = 0
    while t < t_end:
        # Weak form of the heat equation
        a = (u - u0) / time_step * v * dx + k * inner(grad(u), grad(v)) * dx
        
        # Solve the Heat equation (one time step)
        solve(a == 0, u, bc)
        
        # Advance time in exact solution
        t += time_step
        ue.t = t
        
        if step % 5 == 0:
            # Plot solution at current time step
            fig = plt.figure()
            plot(u, fig=fig)
            plt.show()
            
            # Compute error in L2 norm
            error_L2 = errornorm(ue, u, 'L2')
            # Print error
            print(error_L2)
        
        # Shift to next time step
        u0 = project(u, V) 
        step += 1
