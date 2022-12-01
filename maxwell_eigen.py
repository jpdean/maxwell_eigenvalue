# Based on [1]

# References:
# [1]: https://fenicsproject.org/olddocs/dolfin/latest/python/demos/maxwell-eigenvalues/demo_maxwell-eigenvalues.py.html
# [2]: https://bitbucket.org/fenics-project/dolfin/src/master/dolfin/la/SLEPcEigenSolver.cpp
# [3]: https://slepc.upv.es/slepc4py-current/docs/apiref/slepc4py.SLEPc-module.html

import dolfinx
from slepc4py import SLEPc
from ufl import dx, curl, inner, TrialFunction, TestFunction
import numpy as np
from dolfinx.fem import (dirichletbc, Function, FunctionSpace, form,
                         VectorFunctionSpace, locate_dofs_topological)
from mpi4py import MPI
from dolfinx.fem.petsc import assemble_matrix
from petsc4py import PETSc
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
import sys


def par_print(string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def eigenvalues(n_eigs, shift, V, bcs):
    # Define problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = form(inner(curl(u), curl(v)) * dx)
    b = form(inner(u, v) * dx)

    # Assemble matrices
    A = assemble_matrix(a, bcs)
    A.assemble()
    # Zero rows of boundary DOFs of B. See [1]
    B = assemble_matrix(b, bcs, diagonal=0.0)
    B.assemble()

    # Create SLEPc Eigenvalue solver
    eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
    eps.setOperators(A, B)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setWhichEigenpairs(eps.Which.TARGET_MAGNITUDE)
    eps.setTarget(shift)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(shift)

    eps.setDimensions(n_eigs, PETSc.DECIDE, PETSc.DECIDE)
    eps.setFromOptions()
    eps.solve()

    its = eps.getIterationNumber()
    par_print(f"Number of iterations: {its}")

    eps_type = eps.getType()
    par_print(f"Solution method: {eps_type}")

    n_ev, n_cv, mpd = eps.getDimensions()
    par_print(f"Number of requested eigenvalues: {n_ev}")

    tol, max_it = eps.getTolerances()
    par_print(f"Stopping condition: tol={tol}, maxit={max_it}")

    n_conv = eps.getConverged()
    par_print(f"Number of converged eigenpairs: {n_conv}")

    computed_eigenvalues = []
    for i in range(min(n_conv, n_eigs)):
        lmbda = eps.getEigenvalue(i)
        computed_eigenvalues.append(np.round(np.real(lmbda), 1))
    return np.sort(computed_eigenvalues)


def boundary(x):
    lr = boundary_lr(x)
    tb = boundary_tb(x)
    return np.logical_or(lr, tb)


def boundary_lr(x):
    return np.logical_or(np.isclose(x[0], 0.0),
                         np.isclose(x[0], np.pi))


def boundary_tb(x):
    return np.logical_or(np.isclose(x[1], 0.0),
                         np.isclose(x[1], np.pi))


def print_eigenvalues(mesh):
    # Nédélec
    V_nedelec = FunctionSpace(mesh, ("N1curl", 1))

    # Set boundary DOFs to 0 (u x n = 0 on \partial \Omega).
    ud_nedelec = Function(V_nedelec)
    f_dim = mesh.topology.dim - 1
    boundary_facets = locate_entities_boundary(mesh, f_dim, boundary)
    boundary_dofs_nedelec = locate_dofs_topological(
        V_nedelec, f_dim, boundary_facets)
    bcs_nedelec = [dirichletbc(ud_nedelec, boundary_dofs_nedelec)]

    # Solve Maxwell eigenvalue problem
    eigenvalues_nedelec = eigenvalues(n_eigs, shift, V_nedelec, bcs_nedelec)

    # Lagrange
    V_vec_lagrange = VectorFunctionSpace(mesh, ("Lagrange", 1))
    V_lagrange = FunctionSpace(mesh, ("Lagrange", 1))

    # Zero function
    ud_lagrange = Function(V_lagrange)
    # Must constrain horizontal DOFs on horizontal faces and vertical DOFs
    # on vertical faces
    boundary_facets_tb = locate_entities_boundary(mesh, f_dim, boundary_tb)
    boundary_facets_lr = locate_entities_boundary(mesh, f_dim, boundary_lr)
    dofs_tb = locate_dofs_topological(
        (V_vec_lagrange.sub(0), V_lagrange), f_dim, boundary_facets_tb)
    dofs_lr = locate_dofs_topological(
        (V_vec_lagrange.sub(1), V_lagrange), f_dim, boundary_facets_lr)
    bcs_lagrange = [dirichletbc(ud_lagrange, dofs_tb, V_vec_lagrange.sub(0)),
                    dirichletbc(ud_lagrange, dofs_lr, V_vec_lagrange.sub(1))]

    # Solve Maxwell eigenvalue problem
    eigenvalues_lagrange = eigenvalues(
        n_eigs, shift, V_vec_lagrange, bcs_lagrange)

    # Print results
    np.set_printoptions(formatter={'float': '{:5.1f}'.format})
    eigenvalues_exact = np.sort(np.array([float(m**2 + n**2)
                                          for m in range(6)
                                          for n in range(6)]))[1:13]
    par_print(f"Exact    = {eigenvalues_exact}")
    par_print(f"Nédélec  = {eigenvalues_nedelec}")
    par_print(f"Lagrange = {eigenvalues_lagrange}")


# Number of element in each direction
n = 40
# Number of eigernvalues to compute
n_eigs = 12
# Find eigenvalues near
shift = 5.5

comm = MPI.COMM_WORLD

par_print("Right diagonal mesh:")
points = ((0.0, 0.0), (np.pi, np.pi))
mesh = create_rectangle(
    comm,
    points, (n, n),
    CellType.triangle, dolfinx.mesh.GhostMode.none,
    diagonal=dolfinx.mesh.DiagonalType.right)
print_eigenvalues(mesh)

par_print("\nCrossed diagonal mesh:")
mesh = create_rectangle(
    comm,
    points, (n, n),
    CellType.triangle, dolfinx.mesh.GhostMode.none,
    diagonal=dolfinx.mesh.DiagonalType.crossed)
print_eigenvalues(mesh)
