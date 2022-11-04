#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dolfin import *
import numpy as np
parameters["form_compiler"]["representation"] = 'quadrature'
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)


# In[ ]:


# elastic parameters
E = Constant(70e3)
nu = Constant(0.3)
lmbda = E*nu/(1+nu)/(1-2*nu)
mu = E/2./(1+nu)
sig0 = Constant(250.)  # yield strength
Et = E/100.  # tangent modulus
H = E*Et/(E-Et)  # hardening modulus

Re, Ri = 1.3, 1.   # external/internal radius
mesh = Mesh("thick_cylinder.xml")
facets = MeshFunction("size_t", mesh, "thick_cylinder_facet_region.xml")
ds = Measure('ds')[facets]


# In[ ]:


deg_u = 2
deg_stress = 2
V = VectorFunctionSpace(mesh, "CG", deg_u)
We = VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
W = FunctionSpace(mesh, We)
W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W0 = FunctionSpace(mesh, W0e)


# In[ ]:


sig = Function(W)
sig_old = Function(W)
n_elas = Function(W)
beta = Function(W0)
p = Function(W0, name="Cumulative plastic strain")
u = Function(V, name="Total displacement")
du = Function(V, name="Iteration correction")
Du = Function(V, name="Current increment")
v = TrialFunction(V)
u_ = TestFunction(V)


# In[ ]:


bc = [DirichletBC(V.sub(1), 0, facets, 1), DirichletBC(V.sub(0), 0, facets, 3)]


n = FacetNormal(mesh)
q_lim = float(2/sqrt(3)*ln(Re/Ri)*sig0)
loading = Expression("-q*t", q=q_lim, t=0, degree=2)

def F_ext(v):
    return loading*dot(n, v)*ds(4)


# In[ ]:


def eps(v):
    e = sym(grad(v))
    return as_tensor([[e[0, 0], e[0, 1], 0],
                      [e[0, 1], e[1, 1], 0],
                      [0, 0, 0]])
def sigma(eps_el):
    return lmbda*tr(eps_el)*Identity(3) + 2*mu*eps_el
def as_3D_tensor(X):
    return as_tensor([[X[0], X[3], 0],
                      [X[3], X[1], 0],
                      [0, 0, X[2]]])


# In[ ]:


ppos = lambda x: (x+abs(x))/2.
def proj_sig(deps, old_sig, old_p):
    sig_n = as_3D_tensor(old_sig)
    sig_elas = sig_n + sigma(deps)
    s = dev(sig_elas)
    sig_eq = sqrt(3/2.*inner(s, s))
    f_elas = sig_eq - sig0 - H*old_p
    dp = ppos(f_elas)/(3*mu+H)
    n_elas = s/sig_eq*ppos(f_elas)/f_elas
    beta = 3*mu*dp/sig_eq
    new_sig = sig_elas-beta*s
    return as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]),            as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]),            beta, dp


# In[ ]:


def sigma_tang(e):
    N_elas = as_3D_tensor(n_elas)
    return sigma(e) - 3*mu*(3*mu/(3*mu+H)-beta)*inner(N_elas, e)*N_elas-2*mu*beta*dev(e)


# In[ ]:


metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
dxm = dx(metadata=metadata)

a_Newton = inner(eps(v), sigma_tang(eps(u_)))*dxm
res = -inner(eps(u_), as_3D_tensor(sig))*dxm + F_ext(u_)


# In[ ]:


def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dxm
    b_proj = inner(v, v_)*dxm
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return


# In[ ]:


file_results = XDMFFile("plasticity_results.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
P0 = FunctionSpace(mesh, "DG", 0)
p_avg = Function(P0, name="Plastic strain")


# In[ ]:


Nitermax, tol = 200, 1e-8  # parameters of the Newton-Raphson procedure
Nincr = 20
load_steps = np.linspace(0, 1.1, Nincr+1)[1:]**0.5
results = np.zeros((Nincr+1, 2))
for (i, t) in enumerate(load_steps):
    loading.t = t
    print("Load multiplier: ", t)
    A, Res = assemble_system(a_Newton, res, bc)
    nRes0 = Res.norm("l2")
    nRes = nRes0
    Du.interpolate(Constant((0, 0)))
    print("Increment:", str(i+1))
    print("   Initial Residual:", nRes)

    niter = 0
    while nRes/nRes0 > tol and niter < Nitermax:
        solve(A, du.vector(), Res, "mumps")
        # in the first few elastic steps, displacement corr shoul be zero
        # after the first iteration
        Du.assign(Du+du)
        print("Displacement Increment Norm: ", norm(Du.vector(), "l2"))
        print("Displacement Correction Norm: ", norm(du.vector(), "l2"))
        deps = eps(Du)
        sig_, n_elas_, beta_, dp_ = proj_sig(deps, sig_old, p)
        local_project(sig_, W, sig)
        local_project(n_elas_, W, n_elas)
        local_project(beta_, W0, beta)
        A, Res = assemble_system(a_Newton, res, bc)
        nRes = Res.norm("l2")
        print("    Residual:", nRes)
        niter += 1
    u.assign(u+Du)
    p.assign(p+local_project(dp_, W0))
    sig_old.assign(sig)
    file_results.write(u, t)
    p_avg.assign(project(p, P0))
    file_results.write(p_avg, t)
    results[i+1, :] = (u(Ri, 0)[0], t)


# In[ ]:

import matplotlib.pyplot as plt
plt.plot(results[:, 0], results[:, 1], "-o")
plt.xlabel("Displacement of inner boundary")
plt.ylabel(r"Applied pressure $q/q_{lim}$")
plt.show()


# In[ ]:
