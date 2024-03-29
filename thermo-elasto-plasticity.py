#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
# ---


import argparse
import time
import sys
import numpy as np
import ufl
import basix
import dolfinx
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD


def mpi_print(msg):
    if comm.rank == 0:
        print(f"{msg}")


def walltime(func):
    def wrapper(*list_args, **keyword_wargs):
        start_time = time.time()
        func(*list_args, **keyword_wargs)
        end_time = time.time()
        time_elapsed = end_time - start_time
        return time_elapsed
    return wrapper


# Set numpy printing format
np.random.seed(0)
np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=10)

# Manage arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_oris', type=int, default=20)
parser.add_argument('--num_grains', type=int, default=40000)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--domain_x', type=float, help='Unit: mm', default=2.0)
parser.add_argument('--domain_y', type=float, help='Unit: mm', default=2.0)
parser.add_argument('--domain_z', type=float, help='Unit: mm', default=2.0)
parser.add_argument('--dt', type=float, help='Unit: s', default=2e-4)
parser.add_argument('--T_melt', type=float, help='Unit: K', default=1700.)
parser.add_argument('--T_ambient', type=float, help='Unit: K', default=298.)
parser.add_argument('--rho', type=float, help='Unit: kg/mm^3', default=7.68e-6)
parser.add_argument('--c_p', type=float, help='Unit: J/(kg*K)', default=625.)
parser.add_argument('--alpha_V', type=float, help='Unit: mm/K', default=1.61e-5)
parser.add_argument('--Young_mod', type=float, help='Unit: MPa', default=510e4)
parser.add_argument('--power', type=float, help='Unit: W', default=120.)
parser.add_argument('--power_fraction',
                    type=float, help='Unit: None', default=0.39)
parser.add_argument('--r_beam', type=float, help='Unit: mm', default=0.4)
parser.add_argument('--emissivity', type=float, help='Unit:', default=0.2)
parser.add_argument('--SB_constant',
                    type=float, help='Unit: W/(mm^2*K^4)', default=5.67e-14)
parser.add_argument('--h_conv',
                    type=float, help='Unit: W/(mm^2*K)', default=1e-4)
parser.add_argument('--kappa_T',
                    type=float, help='Unit: W/(mm*K)', default=3.4e-2)
parser.add_argument('--write_sol_interval',
                    type=int, help='interval of writing solutions to file',
                    default=500)

args = parser.parse_args("")

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

mpi_print(args)


@walltime
def simulation():
    case_name = 'mechanical'
    def mpi_print(msg):
        if comm.rank == 0:
            print(f"{msg}")

    ambient_T = args.T_ambient
    rho = args.rho
    Cp = args.c_p
    k = args.kappa_T
    h = args.h_conv
    eta = args.power_fraction
    r = args.r_beam
    P = args.power
    dt = args.dt

    x0 = 0.5*args.domain_x
    y0 = 0.5*args.domain_y
    z0 = args.domain_z

    simulation_t = 1e-2#100*dt
    total_t = 1e-2 #100*dt
    vel = 0 #0.6*args.domain_x/total_t
    ts = np.arange(0., simulation_t + dt, dt)
    mpi_print(f"total time steps = {len(ts)}")
    ele_size = 0.1

    Nx, Ny, Nz = round(args.domain_x/ele_size), round(args.domain_y/ele_size), round(args.domain_z/ele_size)

    mpi_print(f"Nx = {Nx}, Ny = {Ny}, Nz = {Nz}")

    mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [np.array([0., 0., 0.]), np.array([args.domain_x, args.domain_y, args.domain_z])],
                                   [Nx, Ny, Nz], cell_type=dolfinx.mesh.CellType.hexahedron)  # cell_type=mesh.CellType.hexahedron/tetrahedron

    # pprint(dir(mesh.geometry))
    print(f"Total number of local mesh vertices {len(mesh.geometry.x)}" )


    def bottom(x):
        return np.isclose(x[2], 0.)

    def top(x):
        return np.isclose(x[2], args.domain_z)
        
    def x_y(x):
        return np.logical_and(np.logical_and(np.isclose(x[0], 2.), np.isclose(x[1], 1.)), np.isclose(x[2], 2.))

    def fixed(x):
        return np.logical_and(np.logical_and(np.isclose(x[0], 2.), np.isclose(x[1], 1.)), np.isclose(x[2], 0.))

    fdim = mesh.topology.dim - 1
    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom)
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top)

    mpi_print(f"bottom_facets.shape = {bottom_facets.shape}")

    marked_facets = np.hstack([bottom_facets, top_facets])
    marked_values = np.hstack([np.full(len(bottom_facets), 1, dtype=np.int32), np.full(len(top_facets), 2, dtype=np.int32)])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(mesh, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

    deg_u = 2
    deg_stress = 2
    degree_T = 1

    # "quadrature_degree": 2 means that use 8 integrations points for a hexahedron element
    metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
    ds = ufl.Measure('ds', domain=mesh, subdomain_data=facet_tag, metadata=metadata)
    dxm = ufl.Measure('dx', domain=mesh, metadata=metadata)
    normal = ufl.FacetNormal(mesh)
    quadrature_points, wts = basix.make_quadrature(basix.CellType.hexahedron, deg_stress)


    P0_ele = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
    P0 = fem.FunctionSpace(mesh, P0_ele)
    p_avg = fem.Function(P0, name="Plastic_strain")
    strain_xx = fem.Function(P0, name="strain_xx")
    strain_yy = fem.Function(P0, name="strain_yy")
    strain_zz = fem.Function(P0, name="strain_zz")
    stress_xx = fem.Function(P0, name="stress_xx")
    stress_xx = fem.Function(P0, name="stress_xx")
    stress_yy = fem.Function(P0, name="stress_yy")
    stress_zz = fem.Function(P0, name="stress_zz")
    phase_avg = fem.Function(P0, name="phase")


    V_ele = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=degree_T)
    V = fem.FunctionSpace(mesh, V_ele)

    U_ele = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=deg_u)
    U = fem.FunctionSpace(mesh, U_ele)

    # W_ele = ufl.TensorElement("DG", mesh.ufl_cell(), 0)
    # W = fem.FunctionSpace(mesh, W_ele)
    # W0_ele = ufl.FiniteElement("DG", mesh.ufl_cell(), 0)
    # W0 = fem.FunctionSpace(mesh, W0_ele)

    # W_ele = ufl.TensorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default', symmetry=True)
    W_ele = ufl.TensorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
    W = fem.FunctionSpace(mesh, W_ele)
    W0_ele = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
    W0 = fem.FunctionSpace(mesh, W0_ele)
    
    
    def l2_projection(v, V):
        dv = ufl.TrialFunction(V)
        v_ = ufl.TestFunction(V)
        a_proj = ufl.inner(dv, v_)*dxm
        b_proj = ufl.inner(v, v_)*dxm
        problem = fem.petsc.LinearProblem(a_proj, b_proj, petsc_options={"ksp_type": "bicg", "pc_type": "jacobi"})
        u = problem.solve()
        return u


    def quad_interpolation(v, V):
        '''
        See https://github.com/FEniCS/dolfinx/issues/2243
        '''
        u = fem.Function(V)
        e_expr = fem.Expression(v, quadrature_points)
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)
        e_eval = e_expr.eval(cells)

        with u.vector.localForm() as u_local:
            u_local.setBlockSize(u.function_space.dofmap.bs)
            u_local.setValuesBlocked(V.dofmap.list.array, e_eval, addv=PETSc.InsertMode.INSERT)

        return u


    def ini_T(x):
        return np.full(x.shape[1], ambient_T)

    dT = fem.Function(V)

    T_crt = fem.Function(V)
    T_crt.interpolate(ini_T)
    T_pre = fem.Function(V)
    T_pre.interpolate(ini_T)
    T_old = fem.Function(V, name='T')
    T_old.interpolate(ini_T)

    T_trial = ufl.TrialFunction(V) 
    T_test = ufl.TestFunction(V)

    phase = fem.Function(V, name='phase')
    alpha_V = fem.Function(V)
    E = fem.Function(V)

    # alpha_V.x.array[:] = 1e-5
    # E.x.array[:] = args.Young_mod

    nu = 0.3
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2./(1+nu)
    sig0 = 50.
    Et = E/100.  
    H = E*Et/(E-Et)  


    sig = fem.Function(W)
    # Something like "Cumulative plastic strain" may cause an error due to the space - probably a bug of dolfinx
    cumulative_p = fem.Function(W0, name="Cumulative_plastic_strain")
    u = fem.Function(U, name="Total_displacement")
    du = fem.Function(U, name="Iteration_correction")
    Du = fem.Function(U, name="Current_increment")
    v = ufl.TrialFunction(U)
    u_ = ufl.TestFunction(U)

    mpi_print(f"facet_tag.dim = {facet_tag.dim}")
    
    U_x, submap = U.sub(0).collapse()
    U_y, submap2 = U.sub(1).collapse()
    
    x_y_dof_x = fem.locate_dofs_geometrical((U.sub(0), U_x), x_y)
    x_y_dof_y = fem.locate_dofs_geometrical((U.sub(1), U_y), x_y)
    
    fixed_dof = fem.locate_dofs_geometrical(U, fixed)
    
    def disp_fixed(x):
        return np.full(x.shape[1], 0)
        
    fixed_disp_x = fem.Function(U_x)
    #fixed_disp_x.interpolate(disp_fixed)

    fixed_disp_y = fem.Function(U_y)
    #fixed_disp_y.interpolate(disp_fixed)

    x_y_bcx = fem.dirichletbc(fixed_disp_x, x_y_dof_x, U.sub(0))
    x_y_bcy = fem.dirichletbc(fixed_disp_y, x_y_dof_y, U.sub(1))
    fixed_bc = fem.dirichletbc(PETSc.ScalarType((0., 0., 0.)), fixed_dof, U)
    
    bcs_u = [x_y_bcx, x_y_bcy, fixed_bc]
   
    #bottom_dofs_u = fem.locate_dofs_topological(U, facet_tag.dim, bottom_facets)
    #bcs_u = [fem.dirichletbc(PETSc.ScalarType((0., 0., 0.)), bottom_dofs_u, U)]

    def eps(v):
        e = ufl.sym(ufl.grad(v))
        return e

    def sigma(eps_el):
        return lmbda*ufl.tr(eps_el)*ufl.Identity(3) + 2*mu*eps_el

    deps = eps(Du)

    def thermal_strain():
        # alpha_V = 1e-5
        return alpha_V*dT*ufl.Identity(3)

    ppos = lambda x: (x + abs(x))/2.
    heaviside = lambda x: ufl.conditional(ufl.gt(x, 0.), 1., 0.)

    def proj_sig():
        EPS = 1e-10
        d_eps_T = thermal_strain()
        sig_elas = sig + sigma(deps - d_eps_T)
        s = ufl.dev(sig_elas)
        sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
        f_elas = sig_eq - sig0 - H*cumulative_p
        dp = ppos(f_elas)/(3*mu+H)
        # Prevent divided by zero error
        # The original example (https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html)
        # didn't consider this, and can cause nan error in the solver.
        n_elas = s/(sig_eq + EPS)*heaviside(f_elas)
        beta = 3*mu*dp/(sig_eq + EPS)
        new_sig = sig_elas - beta*s
        return new_sig, n_elas, beta, dp

    def sigma_tang(e):
        return sigma(e) - 3*mu*(3*mu/(3*mu+H)-beta)*ufl.inner(n_elas, e)*n_elas  -2*mu*beta*ufl.dev(e)


    # If theta = 0., we recover implicit Eulear; if theta = 1., we recover explicit Euler; theta = 0.5 seems to be a good choice.
    theta = 0.5
    T_rhs = theta*T_pre + (1 - theta)*T_trial

    bottom_dofs_T = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.indices[facet_tag.values==1])
    bcs_T = [] #[fem.dirichletbc(PETSc.ScalarType(ambient_T), bottom_dofs_T, V)]

    x = ufl.SpatialCoordinate(mesh)
    crt_time = fem.Constant(mesh, PETSc.ScalarType(0.))

    q_laser = 2*P*eta/(np.pi*r**2) * ufl.exp(-2*((x[0] - x0 - vel*crt_time)**2 + ((x[1] - y0)**2) + (x[2] - z0)) / r**2) #* ufl.conditional(ufl.gt(crt_time, total_t), 0., 1.)
    # q_laser = 2*P*eta/(np.pi*r**2) * ufl.exp(-2*((x[0] - x0 - vel*crt_time)**2 + (x[1] - y0)**2) / r**2) * heaviside(total_t - crt_time.value)


    q_convection = h * (T_rhs - ambient_T)
    res_T = rho*Cp/dt*(T_trial - T_pre) * T_test * dxm + k * ufl.dot(ufl.grad(T_rhs), ufl.grad(T_test)) * dxm \
                - q_laser * T_test * ds(2) #- q_convection * T_test * ds


    new_sig, n_elas, beta, dp = proj_sig()

    # ufl diff might be used to automate the computation of tangent stiffness tensor
    res_u_lhs = ufl.inner(eps(v), sigma_tang(eps(u_)))*dxm
    res_u_rhs = -ufl.inner(new_sig, eps(u_))*dxm

    problem_T = fem.petsc.LinearProblem(ufl.lhs(res_T), ufl.rhs(res_T), bcs=bcs_T, petsc_options={"ksp_type": "bcgs", "pc_type": "jacobi"})

    problem_u = fem.petsc.LinearProblem(res_u_lhs, res_u_rhs, bcs=bcs_u, petsc_options={"ksp_type": "bcgs", "pc_type": "jacobi"})

    def update_modului():
        # 0: powder, 1: liquid, 2: solid 
        T_array = T_crt.x.array

        powder_to_liquid = (phase.x.array == 0) & (T_array > args.T_melt)
        liquid_to_solid = (phase.x.array == 1) & (T_array < args.T_melt)

        phase.x.array[powder_to_liquid] = 1
        phase.x.array[liquid_to_solid] = 2

        E.x.array[(phase.x.array == 0) | (phase.x.array == 1)]  = args.Young_mod # 1e-2*args.Young_mod
        E.x.array[phase.x.array == 2] = args.Young_mod

        alpha_V.x.array[(phase.x.array == 0) | (phase.x.array == 1)] = args.alpha_V #0.
 
        alpha_V.x.array[phase.x.array == 2] = args.alpha_V
  
    def write_sol(file, step):
        file.write_function(T_old, step)
        file.write_function(u, step)
        file.write_function(p_avg, step)
        file.write_function(strain_xx, step)
        file.write_function(strain_yy, step)
        file.write_function(strain_zz, step)
        
        file.write_function(stress_xx, step)
        file.write_function(stress_yy, step) 
        file.write_function(stress_zz, step)         
        #file.write_function(phase_avg, step)

    xdmf_file = dolfinx.io.XDMFFile(mesh.comm, f'{case_name}.xdmf', 'w')

    xdmf_file.write_mesh(mesh)
    write_sol(xdmf_file, 0)

    plastic_inverval = 1

    for i in range(len(ts) - 1):
    # for i in range(20):

        crt_time.value = theta*ts[i] + (1 - theta)*ts[i + 1]

        update_modului()

        T_crt = problem_T.solve()
 
        T_pre.x.array[:] = T_crt.x.array

        # print(f"min T = {np.min(np.array(T_pre.x.array))}")
        # print(f"max T = {np.max(np.array(T_pre.x.array))}\n")
        
        #print(f"number of powder = {np.sum(phase.x.array == 0)}, liquid = {np.sum(phase.x.array == 1)}, solid = {np.sum(phase.x.array == 2)}")

        if (i + 1) % plastic_inverval == 0:
        
            mpi_print(f"At temperature iteration step {i + 1}/{len(ts) - 1}")
            mpi_print(f"\ttime = {ts[i + 1]}")
            
            T_crt_array = np.array(T_crt.x.array)
            T_crt_array = np.where(T_crt_array < args.T_ambient, args.T_ambient, T_crt_array)
            T_crt_array = np.where(T_crt_array > args.T_melt, args.T_melt, T_crt_array)
            T_old_array = np.array(T_old.x.array)
            T_old_array = np.where(T_old_array < args.T_ambient, args.T_ambient, T_old_array)
            T_old_array = np.where(T_old_array > args.T_melt, args.T_melt, T_old_array)
            dT.x.array[:] = T_crt_array - T_old_array

            Du.x.array[:] = 0.

            niter = 0
            nRes = 1.
            tol = 1e-8

            while nRes > tol or niter < 1:
                du = problem_u.solve()
                Du.x.array[:] = Du.x.array + du.x.array

                nRes = problem_u.b.norm(1)
                mpi_print(f"\tb norm: {nRes}")
                niter += 1
                
            mpi_print(f"\tTook plastic iteration steps: {niter}")
            mpi_print(f"\tdu norm = {np.linalg.norm(du.x.array)}")

            u.x.array[:] = u.x.array + Du.x.array

            sig.x.array[:] = quad_interpolation(new_sig, W).x.array
            mpi_print(f"\tsig norm = {np.linalg.norm(sig.x.array)}")

            cumulative_p.x.array[:] = cumulative_p.x.array + quad_interpolation(dp, W0).x.array

            # Remark: Can we do interpolation here?
            # p_avg.interpolate(fem.Expression(cumulative_p, P0.element.interpolation_points))
            p_avg.x.array[:] = l2_projection(cumulative_p, P0).x.array
            strain_xx.x.array[:] = l2_projection(ufl.grad(u)[0, 0], P0).x.array
            strain_yy.x.array[:] = l2_projection(ufl.grad(u)[1, 1], P0).x.array
            strain_zz.x.array[:] = l2_projection(ufl.grad(u)[2, 2], P0).x.array
            
            stress_xx.x.array[:] = l2_projection(sig[0, 0], P0).x.array
            stress_yy.x.array[:] = l2_projection(sig[1, 1], P0).x.array
            stress_zz.x.array[:] = l2_projection(sig[2, 2], P0).x.array
            
            #phase_avg.x.array[:] = l2_projection(phase, P0).x.array

            T_old.x.array[:] = T_crt.x.array

            write_sol(xdmf_file, i + 1)


if __name__ == '__main__':
    simulation()
