import matplotlib.pyplot as plt

from dolfin import Point
from dolfin.fem.assembling import assemble
from dolfin.functions import Expression
from mshr import Circle, Rectangle, generate_mesh
from ufl import ds, dx, one

# Define domain
xmin, xmax = 0.0, 4.0
ymin, ymax = 0.0, 1.0
r = 0.2
x_c, y_c = 0.5, 0.5

# Generate domain and mesh
mesh_resolution = 10
domain = Rectangle(Point(G[0], G[2]), Point(G[1], G[3])) - Circle(Point(x_c, y_c), r)
mesh = generate_mesh(domain, mesh_resolution)

# Visualize the mesh
plt.triplot(mesh)
plt.title("Mesh")

# Create function with value 1 on the interior circle and 0 elsewhere
eps = 1e-5
xi_circle = Expression("(x[0] - xc) * (x[0] - xc) + (y[0] - yc) * (y[0] - yc) < (r + eps) * (r + eps)",
                       xc=x_c, yc=y_c, r=r, eps=eps, domain=mesh, degree=3)
# Compute circle circumference
l = assemble(xi_circle * ds)
print(l)

# Compute the area of the mesh
a = assemble(one * dx)
print(a)
