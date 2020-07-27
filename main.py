from computeexpressions import expression
from initialguesses import guesses
from readdata import read
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

plt.style.use('seaborn-darkgrid')

# biological parameters
n_genes = 17
# experimental setup parameters
n_conditions = 2
n_replicats = 4
n_timesteps = 12
# hypothesis parameters
n_genes_under_TF_control = 15
TF_gene_index = 7
n_ignored_timesteps = 1
# visualization parameters
n_points = 200
conditions = {0: "alginate", 1: "maltose"}

# 1st step: read data
genes_numbers, times, replicatsRTG, replicatsCRTG, replicatsRCTG = read(n_conditions, n_genes, n_genes_under_TF_control, n_replicats, n_timesteps, n_ignored_timesteps)

# 2nd step: make initial guesses for the parameters
replicatsCRT2617 = replicatsCRTG[:, :, :, TF_gene_index]
alpha0, beta0, gamma0, delta0, eta0 = guesses(replicatsCRTG, replicatsCRT2617, n_genes_under_TF_control, n_replicats, times)

# 3rd step: draw figures to actually see something
#timesvisu = np.linspace(times[0], times[-1], n_points, endpoint = False)
theory1, theory2, theory, piecewiseeta = expression(times, np.nanmean(replicatsCRTG[:, : , 0, :], axis=(0, 1)), alpha0, beta0, gamma0, delta0, eta0)
"""
c = 0
figure1 = plt.figure(figsize = (24, 12), dpi = 80)
figure1.canvas.set_window_title('Large operon for condition '+conditions[c])
for g in range(12) :
    plt.subplot(3, 4, g+1)
    plt.plot(times, np.nanmean(replicatsCRTG[c, : , : , g], axis = 0), label = 'measures', marker = 'x')
    plt.plot(times, theory[c, : , g, 0], label = 'model predicted, of which')
    plt.plot(times, theory1[c, : , g, 0], label = '1. own contribution')
    plt.plot(times, theory2[c, : , g, 0], label = '2. production due to TF')
    plt.step(times, piecewiseeta[c, :, 0, -1], where='post', label = 'TF activity')
    plt.title('gene'+str(genes_numbers[g]))
    plt.legend()
    plt.grid(True)
plt.show()
figure2 = plt.figure(figsize = (18, 9), dpi = 80)
figure2.canvas.set_window_title('Small operon and isolated genes for condition '+conditions[0])
for g in range(12, n_genes) :
    plt.subplot(2, 3, g-11)
    plt.plot(times, np.nanmean(replicatsCRTG[c, : , : , g], axis = 0), label = 'measures', marker = 'x')
    plt.plot(times, theory[c, : , g, 0], label = 'model predicted, of which')
    plt.plot(times, theory1[c, : , g, 0], label = '1. own contribution')
    plt.plot(times, theory2[c, : , g, 0], label = '2. production due to TF')
    plt.step(times, piecewiseeta[c, :, 0, -1], where='post', label = 'TF activity')
    plt.title('gene'+str(genes_numbers[g]))
    plt.legend()
    plt.grid(True)
plt.show()
"""

# 4th step: find better values of the parameters
with pm.Model() as Basic_model:

    # Priors for unknown model parameters
    alpha = pm.Uniform('alpha', lower = 0.2*alpha0, upper = 2*alpha0, shape = n_genes_under_TF_control)
    beta  = pm.Uniform('beta' , lower = 0.2*beta0,  upper = 2*beta0,  shape = n_genes_under_TF_control)
    gamma = pm.Uniform('gamma', lower = 0.2*gamma0, upper = 2*gamma0, shape = n_genes_under_TF_control)
    delta = pm.Uniform('delta', lower = 0.2*delta0, upper = 2*delta0, shape = n_genes_under_TF_control)
    eta   = pm.Uniform('eta'  , lower = 0.2*eta0,   upper = 2*eta0,   shape = (n_conditions, n_timesteps-n_ignored_timesteps-1))
#    print('alpha', alpha.type)
#    print('eta[0]', eta[0].type)

    # Expected value of outcome
    mu0 = np.nanmean(replicatsCRTG[ : , : , 0, : ], axis=(0, 1))
#    print('mu0', mu0.shape)
    mu = theory[ : , : , :n_genes_under_TF_control , 0]
#    print('mu', mu.shape)
    mu_flat = mu.flatten()
#    print('mu_flat', mu_flat.shape)

    # Likelihood (sampling distribution) of observations
    sigma = pm.Normal('sigma', sd = 1., shape = (n_conditions * (n_timesteps-n_ignored_timesteps) * n_genes_under_TF_control))
    measures = replicatsRCTG
    measures_flat = measures.reshape(measures.shape[0], -1)
#    print('measures', measures.shape)
#    print('measures_flat', measures_flat.shape)
    Y_obs = pm.Normal('Y_obs', mu = mu_flat, sd = sigma, observed = measures_flat)

    # Draw posterior samples
    trace = pm.sample(init='adapt_diag')

