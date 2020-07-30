from computeexpressions import expression
from initialguesses import guesses
from readdata import read
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

#plt.style.use('seaborn-darkgrid')

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
figure1_title = 'Large operon for condition '+conditions[c]+': initial guesses'
figure1.canvas.set_window_title(figure1_title)
for g in range(12) :
    plt.subplot(3, 4, g+1)
    plt.plot(times, np.nanmean(replicatsCRTG[c, : , : , g], axis = 0), label = 'measures', marker = 'x')
    plt.plot(times, theory [c, : , g, 0], label = 'model predicted, of which')
    plt.plot(times, theory1[c, : , g, 0], label = '1. own contribution')
    plt.plot(times, theory2[c, : , g, 0], label = '2. production due to TF')
    plt.step(times, piecewiseeta[c, :, 0, -1], where='post', label = 'TF activity')
    plt.title('gene'+str(genes_numbers[g]))
    plt.legend()
    plt.grid(True)
plt.savefig(figure1_title+'.png')
plt.show()
figure2 = plt.figure(figsize = (18, 9), dpi = 80)
figure2_title = 'Small operon and isolated genes for condition '+conditions[c]+': initial guesses'
figure2.canvas.set_window_title(figure2_title)
for g in range(12, n_genes) :
    plt.subplot(2, 3, g-11)
    plt.plot(times, np.nanmean(replicatsCRTG[c, : , : , g], axis = 0), label = 'measures', marker = 'x')
    plt.plot(times, theory [c, : , g, 0], label = 'model predicted, of which')
    plt.plot(times, theory1[c, : , g, 0], label = '1. own contribution')
    plt.plot(times, theory2[c, : , g, 0], label = '2. production due to TF')
    plt.step(times, piecewiseeta[c, :, 0, -1], where='post', label = 'TF activity')
    plt.title('gene'+str(genes_numbers[g]))
    plt.legend()
    plt.grid(True)
plt.savefig(figure2_title+'.png')
plt.show()
"""

# 4th step: find better values of the parameters
with pm.Model() as Basic_model:

    # Priors for unknown model parameters
    alpha = pm.Uniform('alpha', lower = 0.2*alpha0, upper = 2.*alpha0, shape = n_genes_under_TF_control)
#    alpha = pm.Gamma('alpha', mu = alpha0, sigma = alpha0, shape = n_genes_under_TF_control)
    beta  = pm.Uniform('beta' , lower = 0.2*beta0,  upper = 2.*beta0,  shape = n_genes_under_TF_control)
#    beta  = pm.Gamma('beta', mu = beta0, sigma = beta0, shape = n_genes_under_TF_control)
    gamma = pm.Uniform('gamma', lower = 0.2*gamma0, upper = 2.*gamma0, shape = n_genes_under_TF_control)
#    gamma = pm.Gamma('gamma', mu = gamma0, sigma = gamma0, shape = n_genes_under_TF_control)
    delta = pm.Uniform('delta', lower = 0.2*delta0, upper = 2.*delta0, shape = n_genes_under_TF_control)
#    delta = pm.Gamma('delta', mu = delta0, sigma = delta0, shape = n_genes_under_TF_control)
    eta   = pm.Uniform('eta'  , lower = 0.2*eta0,   upper = 2.*eta0,   shape = (n_conditions, n_timesteps-n_ignored_timesteps-1))
#    eta   = pm.Gamma('eta'  , mu = eta0,   sigma = eta0,   shape = (n_conditions, n_timesteps-n_ignored_timesteps-1))
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
    sigma = pm.HalfNormal('sigma', sd = 1., shape = (n_conditions * (n_timesteps-n_ignored_timesteps) * n_genes_under_TF_control))
    measures = replicatsRCTG
    measures_flat = measures.reshape(measures.shape[0], -1)
#    print('measures', measures.shape)
#    print('measures_flat', measures_flat.shape)
    Y_obs = pm.Normal('Y_obs', mu = mu_flat, sd = sigma, observed = measures_flat)

    # Draw posterior samples
    trace = pm.sample(init = 'adapt_diag')

# 5th step: draw figures to compare
eta_result = trace.get_values('eta')[-1]
alpha_result = trace.get_values('alpha')[-1]
beta_result  = trace.get_values('beta' )[-1]
gamma_result = trace.get_values('gamma')[-1]
delta_result = trace.get_values('delta')[-1]
theory1, theory2, theory, piecewiseeta = expression(times, np.nanmean(replicatsCRTG[:, : , 0, :], axis=(0, 1)), alpha_result, beta_result, gamma_result, delta_result, eta_result)
for c in range(2) :
    fig, ax = plt.subplots(6, 2, figsize = (12, 21), dpi = 80)
    ax = ax.flatten()
    fig_title = 'Large operon for condition '+conditions[c]+': results'
    fig.canvas.set_window_title(fig_title) # title in the WINDOW
#    fig.suptitle(fig_title) # title in the FIGURE: that's not the same
    for g in range(12) :
        ax[g].plot(times, np.nanmean(replicatsCRTG[c, : , : , g], axis = 0), label = 'measures', marker = 'x')
        ax[g].plot(times, theory [c, : , g, 0], label = 'model predicted, of which')
        ax[g].plot(times, theory1[0, : , g, 0], label = '1. own contribution')
        ax[g].plot(times, theory2[c, : , g, 0], label = '2. production due to TF', linestyle = '--')
        ax[g].step(times, piecewiseeta[c, :, 0, -1], where='post', label = 'TF activity')
        ax[g].set_title('gene'+str(genes_numbers[g]))
        ax[g].legend()
        ax[g].grid(True)
    fig.savefig(fig_title+'.png', bbox_inches = 'tight')
    plt.show()
    fig, ax = plt.subplots(2, 2, figsize = (12, 7), dpi = 80)
    ax = ax.flatten()
    ax[-1].axis('off') # there will be not subplot at the bottom right, so we don't want no axes
    fig_title = 'Small operon and isolated genes for condition '+conditions[c]+': results'
    fig.canvas.set_window_title(fig_title) # title in the WINDOW
#    fig.suptitle(fig_title) # title in the FIGURE: that's not the same
    for g in range(12, n_genes_under_TF_control) :
        ax[g-12].plot(times, np.nanmean(replicatsCRTG[c, : , : , g], axis = 0), label = 'measures', marker = 'x')
        ax[g-12].plot(times, theory [c, : , g, 0], label = 'model predicted, of which')
        ax[g-12].plot(times, theory1[0, : , g, 0], label = '1. own contribution')
        ax[g-12].plot(times, theory2[c, : , g, 0], label = '2. production due to TF', linestyle = '--')
        ax[g-12].step(times, piecewiseeta[c, :, 0, -1], where='post', label = 'TF activity')
        ax[g-12].set_title('gene'+str(genes_numbers[g]))
        ax[g-12].legend()
        ax[g-12].grid(True)
    fig.savefig(fig_title+'.png', bbox_inches = 'tight')
    plt.show()
