#!/usr/bin/env python
# coding: utf-8

# 
# 
# ## Small Strain Von Mises Plasticity Problem 2D

# #### In this notebook, we solve a sample problem in plasticity using a Von Mises yield criterion and Linear Isotropic Hardening
# 
# The key aspects is that this is small strain elastoplasticity and thus the problem can be decomposed (additively) into elastic and plastic parts. The plasticity part contains internal variables for evaluating trial stress, hardening and plastic strain which will then be solved seperately as internal variables compared to the displacements. The **return mapping algorithm** has been used for solving for plastic strain increment and corrections

# In[1]:


from ngsolve import *
from ngsolve.solvers import *
from ngsolve.comp import IntegrationRuleSpace
from ngsolve.fem import MinimizationCF, NewtonCF
from ngsolve.webgui import Draw
import netgen
from netgen.geom2d import CSG2d, Circle, Rectangle
from time import sleep
import numpy as np


# #### Reference Geometry

# In[2]:


geo = CSG2d()

outer_radius = 1.3
inner_radius = 1.0

circle1 = Circle(center=(0,0), radius=outer_radius, bc="outer")
circle2 = Circle(center=(0,0), radius=inner_radius, bc="inner")

rect = Rectangle(pmin=(0,0), pmax=(outer_radius,outer_radius), left="left", bottom="bottom")

geo.Add(rect*(circle1-circle2))

mesh = Mesh(geo.GenerateMesh(maxh=0.05)).Curve(3)
Draw(mesh)


# #### Material Parameters

# In[3]:


E = CF(70000)
nu = CF(0.3)

mu  = E / 2.0 / (1+nu)
lam = E * nu / (1+nu) / (1-2*nu)

Et = E/100.0 #tangent modulus

H = (E*Et)/(E - Et) # hardening modulus

yield_stress = CF(250.0) #  yield strength


# #### Use of Quadrature/Integration Point Space for Internal Variables
# 
# In general, a problem arises in this case when some coefficients in a form are computed by a nonlinear operation elsewhere, and then interpolated and evaluated at a point that differs from where the coefficients were computed
# 
# To avoid involving any interpolation of non-linear expressions throughout the element, the use of a space of the quadrature points on the element is used to to compute values for internal variables (plasticity, hardening, plastic multiplier, stress). It will ensure an optimal convergence rate for the Newton-Raphson method. For this purposes, NGSolve has the notion of an integration rule space. Thus, the problem above can be decomposed into a “global” problem for the displacements and local problems for the internal variables.
# 
# The latter can be solved individually for each quadrature point. However, there is a coupling between the local problems and displacements, which has to be accounted for by what is called “algorithmically consistent linearization”.

# In[4]:


# Create space for displacements (normal vector space)
# This is for the global variables
V = VectorH1(mesh, order = 2, dirichletx="left", dirichlety="bottom") 
u = V.TrialFunction()
v = V.TestFunction()

# Functions defined on the full vector space (global variables)
disp_inc = GridFunction(V)
disp = GridFunction(V)
disp_corr = GridFunction(V)

# A quadrature space for values defined only on Gauss points
fes_quad = IntegrationRuleSpace(mesh, order = 2)

# Extract the integration order and points to be used
irs_dx = dx(intrules=fes_quad.GetIntegrationRules())

# Create a vector space  for total stress (in Voight notation) and is based on Gauss points
# This is essentially the space for internal variables
fes_mat_quad = VectorValued(fes_quad, dim = 4)

# Internal variables
eff_plastic = GridFunction(fes_quad)
plastic_corr = GridFunction(fes_quad)

stress_old = GridFunction(fes_mat_quad)
stress = GridFunction(fes_mat_quad)
yield_normal = GridFunction(fes_mat_quad)

#plotting function
drawfes = H1(mesh, order=2)
pd, qd = drawfes.TnT()
p_strain = GridFunction(drawfes)

# Gradual loading
q_lim = (2.0/sqrt(3.0)*log(outer_radius/inner_radius)*yield_stress)
q = q_lim*specialcf.normal(mesh.dim)

t = Parameter(0.0)

def Load(u):
    loading = t*InnerProduct(q,u)*ds(mesh.Boundaries("inner"))
    return loading


# #### Use of 3D Material Tensor for 2D Problem
# The boundary value problem solved in this tutorial is in plain strain. However, this does not mean that the plastic strains are plain and neither are the elastic ones. Only their sum has vanishing out-of-plane components. The 2D nature of the problem will impose keeping track of the out-of-plane εpzz plastic strain and dealing with representations of stress/strain states including the zz component. Therefore, for simplicity, the material model is implemented in 3d whereas displacements are in 2d.

# In[5]:


# computes strain (linear) from gradient
def Strain(x): 
    _elstrain = Sym(Grad(x)).Compile()
    return CF((_elstrain[0,0], _elstrain[0,1], 0,
               _elstrain[0,1], _elstrain[1,1], 0,
                            0,              0, 0), dims = (3,3))

# computes stress from strain
def Stress(epsilon): 
    return 2*mu*epsilon + lam*Trace(epsilon).Compile()*Id(3)

# computes the deviatoric part of a stress tensor
def Dev(sigma): 
    return sigma - (Trace(sigma).Compile() * Id(3) / 3)

# computes the effective stress according to Von Mises yield criterion
def Eff(dev_sigma): 
    return sqrt((3/2.0)*InnerProduct(dev_sigma,dev_sigma).Compile())

# Takes a 4 dimensional vector (stress voight notation)
# and convert to rank 2 tensor
def as_3D_tensor(X):
    return CF((       X[0], X[3], 0,
                      X[3], X[1], 0,
                       0,    0,  X[2] ), dims =(3,3))

# Only outputs a function value if it is positive
def Is_Positive(x):
    return IfPos(x, x, CF(0.0))


# #### Checking yielding and doing plastic correction

# In[6]:


# If only the positive part of the yield function is taken, 
# then elastic evolution can also be accomodated

def Project_stress(strain_inc, old_sigma, old_p):
    
    # Convert current stress from Voight notation to 3D tensor
    stress_ = as_3D_tensor(old_sigma) 
    
    # Get trial stress from previous stress tensor and new strain inc
    stress_trial = stress_ + Stress(strain_inc) 

    # Computing deviatoric and effective trial stress
    trial_dev = Dev(stress_trial)
    trial_eff = Eff(trial_dev)
    
    # Defining the yield function as well as its positive part (which is 0 for elastic case)
    yield_func = trial_eff - yield_stress - H*old_p 
    pos_yield = Is_Positive(yield_func.Compile())
    
    # Analytical expression for effective plastic strain (in case of linear hardening)
    dp = pos_yield/(3*mu+H)
    
    # Computing normal to yield surface (zero for pure elastic)
    # Computing stress correction from plastic strain (zero for pure elastic)
    yield_normal  = (trial_dev/trial_eff)*(pos_yield)/yield_func
    plastic_corr = 3*mu*dp/(trial_eff) 
    
    new_stress = stress_trial - plastic_corr*trial_dev

    return CF((new_stress[0, 0], new_stress[1, 1], new_stress[2, 2], new_stress[0, 1])), \
           CF((yield_normal[0, 0], yield_normal[1, 1], yield_normal[2, 2], yield_normal[0, 1])), \
           plastic_corr, dp


# #### Linearization of the Problem
# 
# We would need to have an algorithmically consistent linearization based on a coupled field between the internal variables (defined only on Gauss points) and the displacements (defined on the full element) such that the piecewise discontinous stresses can be computed (via a projection) from the space of piecewise continous displacements.

# In[7]:


# Compute the tangent stiffness matrix for the global nonlinear solver
def Jacobian_stress(e):
    Normal = as_3D_tensor(yield_normal)
    return Stress(e) - 3*mu*((3*mu/(3*mu+H))-plastic_corr)*(InnerProduct(Normal, e).Compile())*Normal - 2*mu*plastic_corr*Dev(e)


# In[8]:


a_newton = BilinearForm(V)
a_newton += InnerProduct(Strain(u), Jacobian_stress(Strain(v))).Compile()*irs_dx 

source = LinearForm(V)
source += -InnerProduct(Strain(v), as_3D_tensor(stress)).Compile()*irs_dx + Load(v)


# #### Projection of Internal Variables

# In[9]:


def Local_project(v, V, u=None):
    dv = V.TrialFunction()
    v_ = V.TestFunction()
    
    #Bilinear Form on targeted projection space 
    # (test_function defined in targeted projection space)
    a_proj = BilinearForm(V)
    a_proj += InnerProduct(dv,v_)*irs_dx
    a_proj.Assemble()
    
    #Linear Form on targeted projection space 
    # (using known function from other space)
    b_proj = LinearForm(V)
    b_proj += InnerProduct(v,v_)*irs_dx
    b_proj.Assemble()
    
    # Allows for projection without explicitly 
    # defining the output projected function on projection space
    if u is None:
        u = GridFunction(V)
        u.vec.data = a_proj.mat.Inverse()*b_proj.vec
        return u
    else:
        u.vec.data = a_proj.mat.Inverse()*b_proj.vec
        return


# #### Global Solution
# 
# The newton solver that will conduct corrections to displacements from previous values of stress and plastic strain but the displacement corrections will update the strain increment accordingly until the residual converges to zero and the proper strain increment has been defined to compute the correct trial stress, plastic strain and yield surface normal

# In[10]:


max_iter, tol = 40, 1e-8  # parameters of the Newton-Raphson procedure
loading_inc = 20
load_steps = np.linspace(0, 1.1, loading_inc+1)[1:]**0.5
results = np.zeros((loading_inc+1, 2))


#scene = Draw(disp, mesh)
#scene2 = Draw(eff_plastic, mesh)
#scene3 = Draw(stress.components[0], mesh)

for (i, L) in enumerate(load_steps):
    
    t.Set(L)
    print("Load multiplier: ", t)
    
    a_newton.Assemble()
    source.Assemble()
    
    disp_inc.Interpolate( CF((0,0)) )
    
    nRes0 = Norm(source.vec)#sqrt(InnerProduct(source.vec, source.vec))
    nRes = nRes0

    print("Increment:", str(i+1))
    print("Initial Residual:", nRes)
   
    n_iter = 0
    
    while nRes > tol and n_iter < max_iter:
                
        # Solve for the displacement correction and 
        # use it to correct the displacement increment
        disp_corr.vec.data = a_newton.mat.Inverse(V.FreeDofs()) * source.vec
        disp_inc.vec.data += disp_corr.vec
        
        print("Displacement Increment Norm: ", Norm(disp_inc.vec))
        print("Displacement Correction Norm: ", Norm(disp_corr.vec))
        
        # Compute new iteration of strain increment from displacement corrector
        # This changes the computation of the trial stress and hence the plastic strain
        strain_inc = Strain(disp_inc)
        sig_, n_elas_, beta_, dp_ = Project_stress(strain_inc, stress_old, eff_plastic)

        # Project the quantities back onto their original Quadrature spaces before reuse
        Local_project(sig_, fes_mat_quad, stress)
        Local_project(n_elas_, fes_mat_quad, yield_normal)
        Local_project(beta_, fes_quad, plastic_corr)
        
        # Assemble the system again to compute the new residual
        source.Assemble()
        a_newton.Assemble()
        
        nRes = Norm(source.vec)#sqrt(InnerProduct(source.vec, source.vec))
        
        print("Residual:", nRes)
        
        n_iter += 1
    
    # Update values from final increment calculation
    eff_plastic += Local_project(dp_, fes_quad)
    disp.vec.data += disp_inc.vec
    stress_old = stress

    #scene.Redraw()
    #scene2.Redraw()
    #scene3.Redraw()
    
    results[i+1, :] = (disp(mesh(inner_radius, 0))[0], L)

# #### Quantities
# 
# ##### Hydrostatic Stress ($\boldsymbol\sigma_{m}$) and Deviatoric Stress ($\boldsymbol\sigma^{'}$)
# 
# $$
# \begin{align}
#    \boldsymbol\sigma_{m}  & = \Bigg[\frac{\mathrm{Trace}(\boldsymbol\sigma)}{3}\Bigg]\mathbf{I} \\
#    \boldsymbol\sigma^{'} & = \boldsymbol\sigma - \boldsymbol\sigma_{m}
# \end{align}
# $$
# 
# Effective Stress - Von Mises ($\boldsymbol\sigma_{e}$) and Second Invariant of Deviatoric Stress ($\mathit{J_2}$)
# 
# $$
# \begin{align}
#    \sigma_{e} & = \sqrt{\frac{3}{2}\boldsymbol\sigma^{'}:\boldsymbol\sigma^{'}} \\
#    \mathit{J_2} & = \frac{\sigma_{e}^2}{3}
# \end{align}
# $$
# 
# ##### Effective Plastic Strain Rate
# 
# $$
# \begin{align}
#     \dot{p} & = \sqrt{\frac{2}{3}\boldsymbol{\dot{\varepsilon_p}} : \boldsymbol{\dot{\varepsilon_p}}}\\ 
#    \text{where} \ \mathrm{Trace}(\boldsymbol{\dot{\varepsilon_p}}) & = 0
# \end{align}
# $$
# 
# ##### Normality Hypothesis
# 
# $$
# \begin{align}
#     \mathrm{d}\boldsymbol\varepsilon_p &= \mathrm{d} \lambda \mathbf{n} \\
#     \text{where} \ \mathbf{n} &= \frac{\partial f}{\partial \boldsymbol\sigma}
# \end{align}    
# $$
# 
# With some derivation, the normal ($\mathbf{n}$) can be expressed in terms of effective and deviatoric stress
# 
# $$
# \mathbf{n} = \frac{3}{2}\frac{\boldsymbol\sigma^{'}}{\sigma_{e}}
# $$
# 
# If write the effective plastic strain rate in incremental form,
# 
# $$
# \mathrm{d}p = \sqrt{\frac{2}{3} \mathrm{d}\boldsymbol{\varepsilon_p} : \mathrm{d}\boldsymbol{\varepsilon_p}}
# $$
# 
# Inserting the normality law in for the incremental plastic strain tensor and some manipulation, we can determine,
# 
# $$
# \mathrm{d} \lambda = \mathrm{d}p
# $$
# 
# ##### Consistency Condition
# 
# If,
# 
# $$
# \begin{align}
#     f(\boldsymbol\sigma, p) & = \sigma_e - \sigma_y(p) = 0 \\
#     \Rightarrow \mathrm{d}f & = 0 \\   
# \end{align}
# $$
# 
# Carrying out this differentiation,
# 
# $$
# \begin{align}
#     \frac{\partial f}{\partial \boldsymbol\sigma} : \mathrm{d}\boldsymbol\sigma + \frac{\partial f}{\partial p}:\mathrm{d}p & = 0 \\
#   \Rightarrow \mathbf{n} : \mathrm{d}\boldsymbol\sigma + \frac{\partial f}{\partial p}:\mathrm{d}p & = 0 
# \end{align}
# $$
# 
# ##### Stress increment
# 
# $$
#  \mathrm{d}\boldsymbol\sigma = \mathbf{C}:\mathrm{d} \boldsymbol\varepsilon_{e} = \mathbf{C}:\big(\mathrm{d} \boldsymbol\varepsilon - \mathrm{d} p \mathbf{n}\big)
# $$
# 
# Combining this equation with the consistency condition gives us,
# $$
# \mathrm{d} p = \frac{\mathbf{n}:\mathbf{C}\mathrm{d} \boldsymbol\varepsilon}{\mathbf{n}:\mathbf{C}\mathbf{n} + \frac{\partial f}{\partial p}}
# $$
# 
# where $\frac{\partial f}{\partial p} = H $ if $ \boldsymbol\sigma_{y} = \boldsymbol\sigma_{y0} + H p$ is the linear hardening. This simplifies the above equation to,
# 
# $$
# \mathrm{d} p = \frac{\mathbf{n}:\mathbf{C}\mathrm{d} \boldsymbol\varepsilon}{\mathbf{n}:\mathbf{C}\mathbf{n} + H}
# $$
# 

# #### Starting point
# At the beginning, we start with the strain $\boldsymbol\varepsilon_k$ and stress $\boldsymbol\sigma_k$ at the current step $k$. Moreover, we can split the strain into 2 further parts,
# 
# $$
# \begin{align}
#     \boldsymbol\varepsilon_k = (\boldsymbol\varepsilon_{e})_k + (\boldsymbol\varepsilon_{p})_k
# \end{align}
# $$
# 
# #### Target calculation
# After an increment in the total strain $\Delta \boldsymbol\varepsilon$, we then need the stress $\boldsymbol\sigma_{k+1}$ and strain $\boldsymbol\varepsilon_{k+1}$ for the next step $k + 1$. Thus we need to decompose the strain increment into elastic $\Delta (\boldsymbol\varepsilon_{e})_k$ and plastic $\Delta (\boldsymbol\varepsilon_{p})_k$ parts.
# 
# At the end of the elastic regime, the total strain at the next step will contain both elastic and plastic parts and can be written as
# 
# $$
# \begin{align}
#  \boldsymbol\varepsilon_{k+1} =  \boldsymbol\varepsilon_{k} + \Delta \boldsymbol\varepsilon - \Delta \boldsymbol\varepsilon_{p}
# \end{align}
# $$
# 
# The elastic part of the strain increment is used to calculate the stress increment $\Delta \boldsymbol\sigma$
# 
# $$
# \begin{align}
#     \Delta \boldsymbol\sigma & = \mathbf{C}\Delta \boldsymbol\varepsilon_{e} \\
# \end{align}
# $$
# 
# We calculate the stress (trial) at the next step,
# 
# $$
# \begin{align}
#      \boldsymbol\sigma_{k+1} & = 2 G(\boldsymbol\varepsilon_{e})_{k+1} + \lambda \mathrm{Trace}(\boldsymbol\varepsilon_{e})_{k+1}\mathbf{I} \\
#      \Rightarrow \boldsymbol\sigma & = 2 G(\boldsymbol\varepsilon_{k} + \Delta \boldsymbol\varepsilon - \Delta \boldsymbol\varepsilon_{p}) + \lambda \mathrm{Trace}(\boldsymbol\varepsilon_{k} + \Delta \boldsymbol\varepsilon - \Delta \boldsymbol\varepsilon_{p})\mathbf{I}
# \end{align}
# $$
# 
# Due to the incompressibility constraint, $\mathrm{Trace}(\Delta \boldsymbol\varepsilon_{p}) = 0$, so we can rearrange the above equation to give us,
# 
# #### Hooke's law
# $$
# \begin{align}
#     \Rightarrow \boldsymbol\sigma & = 2 G(\boldsymbol\varepsilon_{k} + \Delta \boldsymbol\varepsilon) + \lambda \mathrm{Trace}(\boldsymbol\varepsilon_{k} + \Delta \boldsymbol\varepsilon)\mathbf{I} - 2 G\Delta \boldsymbol\varepsilon_{p}
# \end{align}
# $$
# 
# The term $2 G(\boldsymbol\varepsilon_{k} + \Delta \boldsymbol\varepsilon) + \lambda \mathrm{Trace}(\boldsymbol\varepsilon_{k} + \Delta \boldsymbol\varepsilon)\mathbf{I}$ is known as the trial stress ($\boldsymbol\sigma_{tr} = \boldsymbol\sigma + \mathbf{C}\Delta \boldsymbol\varepsilon$) and therefore the stress at the next step contains and adjustment from plastic part of the strain,
# 
# $$
# \begin{align}
#     \boldsymbol\sigma_{k+1} = \boldsymbol\sigma_{tr} - 2 G\Delta \boldsymbol\varepsilon_{p}
# \end{align}
# $$
# where $\Delta \boldsymbol\varepsilon_{p}=\Delta p\frac{3}{2}\frac{\boldsymbol\sigma^{'}}{\sigma_{e}}$ and thus 
# 
# $$
# \begin{align}
#     \boldsymbol\sigma_{k+1} = \boldsymbol\sigma_{tr} - 2 G\Delta p\frac{3}{2}\frac{\boldsymbol\sigma^{'}}{\sigma_{e}}
# \end{align}
# $$
# 
# #### Trial stress and plastic correction
# Consider three cases for trial stress,
# 
# $\textbf{Case 1}$: $\boldsymbol\sigma_{k}$ and $\boldsymbol\sigma_{tr}$ lie inside the yield surface:
# 
# In this case, the deformation is elastic and $\Delta \varepsilon_{p} = 0$.
# 
# $\textbf{Case 2}$: $\boldsymbol\sigma_{k}$ lies on the yield surface and $\boldsymbol\sigma_{tr}$ goes outside the yield surface:
# 
# Plastic correction must be done to bring $\boldsymbol\sigma_{k+1}$ to the yield surface
# 
# $\textbf{Case 3}$: $\boldsymbol\sigma_{k}$ lies inside the yield surface and $\boldsymbol\sigma_{tr}$ goes outside the yield surface:
# 
# Plastic correction must be done to bring $\boldsymbol\sigma_{k+1}$ to the yield surface
# 
# Solving,
# 
# $$
# \boldsymbol\sigma = \boldsymbol\sigma_{tr} - 2 G\Delta p\frac{3}{2}\frac{\boldsymbol\sigma^{'}}{\sigma_{e}}
# $$
# 
# requires solving for $\Delta p$ requires six equations (one for each derivative component of stress) but this can be simplified further by expressing the left hand side also in terms of deviatoric stress,
# 
# $$
# \begin{align}
#     \boldsymbol\sigma^{'} + \frac{1}{3}\mathrm{Trace}(\boldsymbol\sigma_m) & = \boldsymbol\sigma_{tr} - 2 G\Delta p\frac{3}{2}\frac{\boldsymbol\sigma^{'}}{\sigma_{e}} \\
# \end{align}
# $$
# 
# Rearranging leads to 
# 
# $$
# \begin{align}
#     \Big(1+3G \frac{\Delta p}{\sigma_{e}} \Big)\boldsymbol\sigma^{'} & = \boldsymbol\sigma_{tr} - \frac{1}{3}\mathrm{Trace}(\boldsymbol\sigma_m) \\
#     \Rightarrow \Big(1+3G \frac{\Delta p}{\sigma_{e}} \Big)\boldsymbol\sigma^{'} & = \boldsymbol\sigma^{'}_{tr}
# \end{align}
# $$
# 
# Multiplying (via double dot product) on both sides by deviatoric stresses and then taking the square root,
# 
# $$
# \begin{align}
#     \Rightarrow \Big(1+3G \frac{\Delta p}{\sigma_{e}} \Big)\boldsymbol\sigma^{'}:\boldsymbol\sigma^{'} & = \boldsymbol\sigma^{'}_{tr}:\boldsymbol\sigma^{'}_{tr} \\
#     \Rightarrow \Big(1+3G \frac{\Delta p}{\sigma_{e}} \Big)\sigma_e & = (\sigma_{tr})_e
# \end{align}
# $$
# 
# This way we can express the effective stress in terms of the trial stress and the effective plastic strain change,
# $$
#     \sigma_e = (\sigma_{tr})_e - 3G\Delta p
# $$
# 
# If we input the previous expression in the definition of the yield function we get,
# 
# $$
# \begin{align}
#     f &= \sigma_e - \sigma_y\\
#     \Rightarrow f &= (\sigma_{tr})_e - 3G\Delta p - \sigma_y
# \end{align}
# $$
# 
# This precisely the non-linear equation we must solve (via Newton-Raphson method) to determine effective plastic strain part of the total strain increment $\Delta p$ i.e.,
# 
# $$ 
# f + \frac{\partial f}{\partial \Delta p} \mathrm{d}\Delta p + \ ...\  = 0
# $$
# 
# Neglecting higher order terms (Taylor approximation),
# 
# 
# $$
# \begin{align}
#     (\sigma_{tr})_e - 3G\Delta p - \sigma_y + (-3G - H)\mathrm{d}\Delta p &= 0\\
# \end{align}
# $$
# 
# Hence, the correction in effective plastic strain change is,
# 
# $$
#     \mathrm{d}\Delta p = \frac{(\sigma_{tr})_e - 3G\Delta p - \sigma_y}{3G + H}
# $$
# 
# Solved iteratively, this will give us the necessary adjustment needed in the effective plastic strain increment $\Delta p$ to bring the trial stress $\boldsymbol\sigma_tr$ back to the yield surface and thus giving as a new value for effective plastic strain $p_{k+1}$ to change the yield surface for next step.
# 
# The plastic strain increment can then be determined from the deviatoric part of the trial stress, its effective value and the recently determined effective plastic strain increment i.e. $\Delta \boldsymbol\varepsilon_{p}=\Delta p\frac{3}{2}\frac{\boldsymbol\sigma^{'}}{\sigma_{e}}$

# #### Jacobian Matrix
# 
# The Jacobian Matrix is a representation of the infinitesmal change in stress with respect to an infinitesmal change in strain and thus controls the stability of the stiffness matrix created for the FEM solution. This is an additional step for the time-discrete evolution of the internal variables (plastic strains and hardening parameters) due to the incremental nature of the variational (and nonlinear) problem.
# 
# The matrix is represented as,
# 
# $$
#     \mathbf{C}_{ep} = \frac{\partial \delta \sigma}{\partial \delta \varepsilon}
# $$
# 
# The exact calculation of this matrix allows for faster convergence of the solver.
# 
# For the simple elastic problem $\sigma = \mathbf{C}\varepsilon$, it is simply the elastic constitutive tensor $\mathbf{C}$.
# 
# Defining the Jacobian Matrix of plastic material is different and for it we start with the decomposition of stress,
# 
# $$
# \begin{align}
#     \boldsymbol\sigma & = \boldsymbol\sigma^{'} + \frac{1}{3}\mathrm{Trace}(\boldsymbol\sigma)\mathbf{I} \\
#     \Rightarrow \delta\boldsymbol\sigma & = \delta  \boldsymbol\sigma^{'} + \frac{1}{3}\mathrm{Trace}(\delta\boldsymbol\sigma)\mathbf{I}
# \end{align}
# $$
# 
# The second part of this represents the hydrostatic stress and can be reformulated using the Bulk Modulus $K$,
# 
# $$
# \begin{align}
#     \Rightarrow \delta\boldsymbol\sigma & = \delta  \boldsymbol\sigma^{'} + K \delta V \\
#     \Rightarrow \delta\boldsymbol\sigma & = \delta  \boldsymbol\sigma^{'} + K \mathrm{Trace}(\delta\boldsymbol\varepsilon_e)\mathbf{I} \\
#     \Rightarrow \delta\boldsymbol\sigma & = \delta  \boldsymbol\sigma^{'} + K \mathrm{Trace}(\delta\boldsymbol\varepsilon-\delta\boldsymbol\varepsilon_p)\mathbf{I}
# \end{align}
# $$
# 
# where $\mathrm{Trace}(\delta\boldsymbol\varepsilon_p) = 0$ and hence 
# 
# $$
# \Rightarrow \delta\boldsymbol\sigma = \delta  \boldsymbol\sigma^{'} + K \mathrm{Trace}(\delta\boldsymbol\varepsilon)\mathbf{I}
# $$
# 
# Now, let's take the definition of the deviatoric trial stress and apply a small perturbation to it,
# 
# $$
# \begin{align}
#     \Big(1+3G \frac{\Delta p}{\sigma_{e}} \Big)\boldsymbol\sigma^{'} & = \boldsymbol\sigma^{'}_{tr}\\
#     \Rightarrow \delta\Bigg[\Big(1+3G \frac{\Delta p}{\sigma_{e}} \Big)\boldsymbol\sigma^{'}\Bigg] &= \delta(\boldsymbol\sigma^{'}_{tr})
# \end{align}
# $$
# Successful application of the product rule and quotient rule of diffrentitation (w.r.t. $\boldsymbol\sigma^{'}, \Delta p $ and $\sigma_{e}$) leads to,
# 
# $$
# \delta\Bigg[\Big(1+3G \frac{\Delta p}{\sigma_{e}} \Big)\boldsymbol\sigma^{'} \Bigg]= \Big(1+3G \frac{\Delta p}{\sigma_{e}} \Big)\delta \boldsymbol\sigma^{'} + 3G \frac{\delta \Delta p}{\sigma_{e}}\boldsymbol\sigma^{'} +  3G \frac{\delta \Delta p}{\sigma^{2}_{e}}\delta \sigma_{e}\boldsymbol\sigma^{'} =\delta(\boldsymbol\sigma^{'}_{tr})
# $$
# 
# We already have $\Delta p = \frac{(\sigma_{e})_{tr}-\sigma_{e}}{3G}$ and its variation $\delta \Delta p$ can be expressed as,
# $$
# \begin{align}
#     \delta \Delta p = \delta \Big(\frac{(\sigma_{e})_{tr}-\sigma_{e}}{3G}\Big)
# \end{align}
# $$
# where $\delta \sigma_{e}$ can be expressed from the variation of the yield function $\delta f = \delta \sigma_{e} - \delta \sigma_{y} = \delta \sigma_{e} - H \delta \Delta p = 0$ and thus $\delta \sigma_{e} = H \delta \Delta p$.
# 
# This will then lead to
# 
# $$
# H \delta \Delta p + 3G \delta \Delta p = \delta (\sigma_{e})_{tr}
# $$
# 
# and thus finally,
# 
# $$
# \delta \Delta p = \frac{\delta (\sigma_{e})_{tr}}{H + 3G}
# $$
# 
# Lastly we need $\delta \sigma_e$ and we reuse the previous equations but this time, replace with the variation $\delta \Delta p$ that we have just determined above and we get,
# 
# $$
# \delta \sigma_e + 3G \frac{\delta (\sigma_e)_{tr}}{H+3G} = \delta (\sigma_e)_{tr}
# $$
# 
# So, $\delta \sigma_e = \frac{H}{H+3G}\delta (\sigma_e)_{tr}$, variations of all three parameters have been determined and we can replace them in the original equation and get the variation in trial deviatoric stress as,
# 
# $$
#     \frac{(\sigma_e)_{tr}}{\sigma_e} \delta \boldsymbol\sigma^{'} + \frac{3G}{\sigma_e} \frac{\delta (\sigma_e)_{tr}}{H + 3G}\boldsymbol\sigma^{'} - \frac{(\sigma_e)_{tr}-\sigma_e}{\sigma^{2}_e} \Big(\frac{H}{H+3G} \Big)\delta(\sigma_e)_{tr}\boldsymbol\sigma^{'} = \delta \boldsymbol{(\sigma^{'})}_{tr}
# $$
# 
# and some manipulation gives us
# 
# $$
#     \frac{(\sigma_e)_{tr}}{\sigma_e} \delta \boldsymbol\sigma^{'} + \frac{\delta(\sigma_e)_{tr}}{\sigma_e} \boldsymbol\sigma^{'} - \frac{(\sigma_e)_{tr}}{\sigma^{2}_e} \Big(\frac{H}{H+3G}\Big) \delta(\sigma_e)_{tr}\boldsymbol\sigma^{'} = \delta \boldsymbol{(\sigma^{'})}_{tr}
# $$
# 
# The variation of trial effective stress $\delta(\sigma_e)_{tr}$,
# 
# $$
# \begin{align}
#     (\sigma_e)_{tr} & = \sqrt{\frac{3}{2}\boldsymbol{(\sigma^{'})}_{tr}:\boldsymbol{(\sigma^{'})}_{tr}} \\
#     \Rightarrow \delta(\sigma_e)_{tr} & = \frac{\frac{3}{2} \Big((\sigma_e)_{tr}: \delta (\sigma_e)_{tr} + \delta (\sigma_e)_{tr}:(\sigma_e)_{tr}\Big)}{2 \sqrt{\frac{3}{2}\boldsymbol{(\sigma^{'})}_{tr}:\boldsymbol{(\sigma^{'})}_{tr}}} \\
#     \Rightarrow \delta(\sigma_e)_{tr} & = \frac{3 \boldsymbol{(\sigma^{'})}_{tr}:\delta \boldsymbol{(\sigma^{'})}_{tr}}{2(\sigma_e)_{tr} }
# \end{align}
# $$
# 
# With this the final expression for variation in trial deviatoric stress becomes,
# 
# $$
#     \frac{(\sigma_e)_{tr}}{\sigma_e} \delta \boldsymbol\sigma^{'} + \frac{3 \boldsymbol{(\sigma^{'})}_{tr}:\delta \boldsymbol{(\sigma^{'})}_{tr}}{2(\sigma_e)_{tr} \sigma_e} \boldsymbol\sigma^{'} - \Big(\frac{H}{H+3G}\Big) \frac{3 \boldsymbol{(\sigma^{'})}_{tr}:\delta \boldsymbol{(\sigma^{'})}_{tr}}{2\sigma^{2}_e}\boldsymbol\sigma^{'} = \delta \boldsymbol{(\sigma^{'})}_{tr}
# $$
# 
# Since the original goal was always to study the variation of deviatoric stress with relation to variation in strain, the above equation must be rearranged to make $\delta \boldsymbol{(\sigma^{'})}$ the subject,
# 
# $$
#     \delta \boldsymbol{(\sigma^{'})} = \frac{3}{2}\Big(\frac{H}{H+3G} - \frac{\sigma_e}{(\sigma_e)_{tr}} \Big)\frac{\boldsymbol{(\sigma^{'})}_{tr}}{(\sigma_e)_{tr}}\frac{\boldsymbol{(\sigma^{'})}_{tr}}{(\sigma_e)_{tr}}:\delta \boldsymbol{(\sigma^{'})}_{tr} + \frac{\sigma_e}{(\sigma_e)_{tr}} \delta \boldsymbol{(\sigma^{'})}_{tr}
# $$
# 
# For simplification, let $Q = \frac{3}{2}\Big(\frac{H}{H+3G} - \frac{\sigma_e}{(\sigma_e)_{tr}} \Big)$ which is a scalar, and expression $\frac{\boldsymbol{(\sigma^{'})}_{tr}}{(\sigma_e)_{tr}} = \boldsymbol{n}$, since it represents the normal to the yield surface. Also, we can express $R = \frac{\sigma_e}{(\sigma_e)_{tr}}$, then the above expression can be expressed as,
# 
# $$
#     \delta \boldsymbol{(\sigma^{'})} = Q \boldsymbol{n}\boldsymbol{n}:\delta \boldsymbol{(\sigma^{'})}_{tr} + R\delta \boldsymbol{(\sigma^{'})}_{tr}
# $$
# 
# Summing up, a relation between $\delta \boldsymbol{(\sigma^{'})}_{tr}$ and $\delta  \boldsymbol\varepsilon$ must be established and for that let us first rewrite the constitutive law for elastic stress tensor,
# 
# $$
# \begin{align}
#     \boldsymbol\sigma = 2G \boldsymbol{\varepsilon_e} + \lambda \text{Trace}(\boldsymbol{\varepsilon_e})\mathbf{I}
# \end{align}
# $$
# 
# For deviatoric stresses, $\text{Trace}(\boldsymbol{\varepsilon_e})=0$,
# 
# $$
# \begin{align}
#     \Rightarrow \boldsymbol{\sigma^{'}} & = 2G \boldsymbol{(\varepsilon^{'}_e)_{tr}} \\
#     \Rightarrow \delta \boldsymbol{\sigma^{'}} & = 2G \delta \boldsymbol{(\varepsilon^{'}_e)_{tr}}\\
#     \Rightarrow \delta \boldsymbol{\sigma^{'}} & = 2G  \big(\delta \boldsymbol{\varepsilon_e} - \frac{1}{3}\text{Trace}(\delta \boldsymbol{\varepsilon_e}) \big)
# \end{align}
# $$
# 
# Equating with previous expression for $\delta \boldsymbol{\sigma^{'}}$,
# 
# $$
#     \delta \boldsymbol{(\sigma^{'})} = 2G Q \boldsymbol{n}\boldsymbol{n}:\delta \boldsymbol{\varepsilon} -  2G Q \boldsymbol{n}\boldsymbol{n}:\frac{1}{3}\text{Trace}(\delta \boldsymbol{\varepsilon})\mathbf{I} + 2G R\delta \boldsymbol{\varepsilon} - 2GR \Big( \frac{1}{3}\text{Trace}(\delta \boldsymbol{\varepsilon})\mathbf{I}\Big)
# $$
# 
# where $2G Q \boldsymbol{n}\boldsymbol{n}:\frac{1}{3}\text{Trace}(\delta \boldsymbol{\varepsilon})\mathbf{I} = 0$, due to the fact the double product of a deviatoric tensor and identity tensor is zero,
# 
# Finally our variation in deviatoric stress is,
# $$
# \delta \boldsymbol{(\sigma^{'})} = 2G Q \boldsymbol{n}\boldsymbol{n}:\delta \boldsymbol{\varepsilon} + 2G R\delta \boldsymbol{\varepsilon} - 2GR \Big( \frac{1}{3}\text{Trace}(\delta \boldsymbol{\varepsilon})\mathbf{I}\Big)
# $$
# 
# Combining this with our general stress variation $\delta\boldsymbol\sigma = \delta  \boldsymbol\sigma^{'} + K \mathrm{Trace}(\delta\boldsymbol\varepsilon)\mathbf{I}$, we get
# 
# $$
# \delta\boldsymbol\sigma = 2G Q \boldsymbol{n}\boldsymbol{n}:\delta \boldsymbol{\varepsilon} + 2G R\delta \boldsymbol{\varepsilon} + (K - \frac{2}{3}GR)\mathrm{Trace}(\delta\boldsymbol\varepsilon)\mathbf{I}
# $$
# 
# Thus, we have effectively established a tangent stiffness matrix for the Newton solver that relates variations in stresses to variations in strains, and can be computed through values of $Q$, $R$ and $n$ which are functions of $H$, $G$, $\boldsymbol\sigma$ and $\boldsymbol{\sigma_{tr}}$.

# #### This entire process can be illustrated with the following algorithm:
# 
# 1. $\textbf{Start of Step:}$ \
#     At the beginning of the step, we know total strain $\boldsymbol\varepsilon_{k}$, total stress $\boldsymbol\sigma_{k}$, the total strain increment $\Delta \boldsymbol\varepsilon $ and the effective plastic strain $p_k$ ($=0$ in the first step)
# 
# 2. $\textbf{Calculate Trial Stress:}$ \
#     $\boldsymbol\sigma_{tr} = \boldsymbol\sigma_k + \mathbf{C}\Delta \boldsymbol\varepsilon$
# 
# 3. $\textbf{Determine Effective Trial Stress:}$ \
#     $(\sigma_{tr})_e = \sqrt{\frac{3}{2}\boldsymbol\sigma^{'}_{tr}:\boldsymbol\sigma^{'}_{tr}}$
#     
# 4. $\textbf{Determine Flow Stress:}$ \
#     $\boldsymbol\sigma_{y}=f(p)=\boldsymbol\sigma_{y0} + H p$
#     
# 5. $\textbf{Determine Trial Yield Function:}$ \
#     $f = (\sigma_{tr})_e - \boldsymbol\sigma_{y}$
#     
# 6. $\textbf{Check the value of Yield Function:}$ \
#     If $f < 0$, the deformation is elastic and there are no plastic strains i.e. $\Delta p = 0$ otherwise Newton method must be used to solve for the plastic strain increment $\Delta p$
#     
# 7. $\textbf{Calculate the Newton iteration for Plastic correction :}$ \
#     $\mathrm{d}\Delta p = \frac{(\sigma_{tr})_e - 3G\Delta p - \sigma_y}{3G + H}$
#     
# 8. $\textbf{Update Plastic Increment:}$ \
#     $\Delta p = \Delta p + \mathrm{d}\Delta p $
#     
# 9. $\textbf{Update Yield Surface:}$ \
#     $\sigma_y = \sigma_{y0} + H p$
#     
# 10. $\textbf{Check plastic strain correction against threshold:}$ \
#     This is to ensure the plastic strain increment is not too large ($\mathrm{d}\delta p < 0$). If it is too large, the Newton solver did not converge
#     
# 11. $\textbf{Compute plastic strain tensor:}$ \
#     $\Delta \boldsymbol\varepsilon_{p} = \Delta p\frac{3}{2}\frac{\boldsymbol\sigma^{'}}{\sigma_{e}}$
#     
# 12. $\textbf{Compute elastic strain tensor:}$ \
#     $\Delta \boldsymbol\varepsilon_{e} = \Delta \boldsymbol\varepsilon - \Delta \boldsymbol\varepsilon_{p}$
#     
# 13. $\textbf{Compute stress increment:}$ \
#     $\Delta \sigma = \mathbf{C}\Delta \boldsymbol\varepsilon_{e}$
#     
# 14. $\textbf{Update variables:}$ \
#     $\boldsymbol\sigma_{k+1} = \boldsymbol\sigma_{k} + \Delta \sigma$ \
#     $p_{k+1} = p_{k} + \Delta p$
#     
# 15. $\textbf{Calculate Jacobian Matrix:}$ \
#     A consisten Jacobian Matrix needs to be computed for calculating the tangent stiffness between stress and strain variation
#     
# 16. $\textbf{Calculate Displacements:}$ 
#     Using Jacobian Matrix, displacements can be calculated
