from computeexpressions import expression
from initialguesses import guesses
from readdata import read
import matplotlib.pyplot as plt
import numpy as np

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
c = 1

# 1st step: read data
genes_numbers, times, replicatsRTG, replicatsCRTG = read(n_conditions, n_genes, n_replicats, n_timesteps, n_ignored_timesteps)
# 2nd step: make initial guesses for the parameters
replicatsCRT2617 = replicatsCRTG[:, :, :, TF_gene_index]
alpha0, beta0, gamma0, delta0, eta0 = guesses(replicatsCRTG, replicatsCRT2617, n_genes, n_replicats, times)
# 3rd step: draw figures to actually see something
timesvisu = np.linspace(times[0], times[-1], n_points, endpoint = False)
figure1 = plt.figure(figsize = (24, 12), dpi = 80)
figure1.canvas.set_window_title('Large operon for condition '+conditions[c])
for g in range(12) :
    theory1, theory2, theory, piecewiseeta = expression(timesvisu, times, np.nanmean(replicatsCRTG[c, : , 0, g]), alpha0[g], beta0[g], gamma0[g], delta0[g], eta0[c])
    plt.subplot(3, 4, g+1)
    plt.plot(times, np.nanmean(replicatsCRTG[c, : , : , g], axis = 0), label = 'measures', marker = 'x')
    plt.plot(timesvisu, theory, label = 'model predicted, of which')
    plt.plot(timesvisu, theory1, label = '1. own contribution')
    plt.plot(timesvisu, theory2, label = '2. production due to TF')
    plt.plot(timesvisu, piecewiseeta[-1], label = 'TF activity')
    plt.title('gene'+str(genes_numbers[g]))
    plt.legend()
    plt.grid(True)
plt.show()
figure2 = plt.figure(figsize = (18, 9), dpi = 80)
figure2.canvas.set_window_title('Small operon and isolated genes for condition '+conditions[c])
for g in range(12, n_genes) :
    theory1, theory2, theory, piecewiseeta = expression(timesvisu, times, np.nanmean(replicatsCRTG[c, : , 0, g]), alpha0[g], beta0[g], gamma0[g], delta0[g], eta0[c])
    plt.subplot(2, 3, g-11)
    plt.plot(times, np.nanmean(replicatsCRTG[c, : , : , g], axis = 0), label = 'measures', marker = 'x')
    plt.plot(timesvisu, theory, label = 'model predicted, of which')
    plt.plot(timesvisu, theory1, label = '1. own contribution')
    plt.plot(timesvisu, theory2, label = '2. production due to TF')
    plt.plot(timesvisu, piecewiseeta[-1], label = 'TF activity')
    plt.title('gene'+str(genes_numbers[g]))
    plt.legend()
    plt.grid(True)
plt.show()
# 4th step: find better values of the parameters
"""
with pm.Model() as Basic_model:

    # Priors for unknown model parameters
    alpha = pm.HalfNormal('alpha', sd = 10, shape = n_genes_under_TF_control)
    beta  = pm.HalfNormal('beta' , sd = 10, shape = n_genes_under_TF_control)
    gamma = pm.HalfNormal('gamma', sd = 10, shape = n_genes_under_TF_control)
    delta = pm.HalfNormal('delta', sd = 10, shape = n_genes_under_TF_control)
    eta   = pm.HalfNormal('eta'  , sd = 10, shape = n_timesteps-n_ignored_timesteps-1)
    print('alpha', alpha.type)
    print('eta[0]', eta[0].type)

    # Expected value of outcome
    mu0      = np.nanmean(exprcrtg[:, :, 0, :], axis = (0, 1))
    print('mu0', mu0.shape)
    mu = expression()[2]
    print('mu', mu.shape)

    # Likelihood (sampling distribution) of observations
    sigma = pm.HalfNormal('sigma', sd = 10, shape = n_genes_under_TF_control)
    print('sigma', sigma.shape)
    Y_obs = pm.Normal('Y_obs', mu = mu, sd = sigma, observed = exprtg[:, 1:, :])

    # Draw 500 posterior samples
    trace = pm.sample(500)
"""
