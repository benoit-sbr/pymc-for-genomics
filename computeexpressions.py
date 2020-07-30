import numpy as np

def expression(times, mu0, alpha, beta, gamma, delta, eta):
    # Docstring of function
    """
    Returns the expression of mu
    Positional arguments:
    times	--- array of times
    mu0		--- expression levels at t = 0
    alpha	---
    beta	---
    gamma	---
    delta	---
    eta		--- array of values for the piecewise-constant function eta
    Keyword arguments:
    """
    # Body of function
    alpha_new = alpha[np.newaxis, np.newaxis, : , np.newaxis]
    beta_new  = beta [np.newaxis, np.newaxis, : , np.newaxis]
    gamma_new = gamma[np.newaxis, np.newaxis, : , np.newaxis]
    delta_new = delta[np.newaxis, np.newaxis, : , np.newaxis]
    tau = (np.linspace(0., times, 200)).transpose()
    tau_new = tau[np.newaxis, :, np.newaxis, : ]
#    print('tau.shape', tau.shape)
#    print('tau', tau)
#    print('tau[-1, :]', tau[-1, :])
    positions = (np.searchsorted(times, tau, side='right') - 1)
    positions_new = positions[np.newaxis, :, np.newaxis, : ]
#    print('positions.shape', positions.shape)
#    print('positions', positions)
#    print('positions[-1, :]', positions[-1, :])
    eta_0     = np.ones((2, 1))
    eta_full  = np.concatenate((eta_0, eta), axis = 1)
#    print('positions[np.newaxis, :, np.newaxis].shape', positions[np.newaxis, :, np.newaxis].shape)
    eta_new   = eta_full[:, positions]
#    print('eta_new.shape', eta_new.shape)
#    print('eta_new', eta_new)
#    print('eta_new[0, -1, :]', eta_new[0, -1, :])
    eta_tau = eta_new[: , : , np.newaxis, : ]
    integrand = np.exp(delta_new*tau_new) * eta_tau / (gamma_new + eta_tau)
#    print('integrand.shape', integrand.shape)
#    print('integrand', integrand)
    integrale = np.trapz(integrand, tau_new)
    integrale_new = integrale[ : , : , : , np.newaxis]
#    print('integrale.shape', integrale.shape)
#    print('mu0.shape', mu0.shape)
#    print('alpha_new.shape', alpha_new.shape)
    times_new = times[np.newaxis, :, np.newaxis, np.newaxis]
#    print('times_new.shape', times_new.shape)
#    print('.shape', .shape)
#    print('.shape', .shape)
    half1st = (mu0 - alpha_new/delta_new) * np.exp(-delta_new*times_new) + alpha_new/delta_new
    half2nd = beta_new * np.exp(-delta_new*times_new) * integrale_new
    result  = half1st + half2nd
    return half1st, half2nd, result, eta_tau
