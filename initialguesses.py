from scipy.optimize import curve_fit
import numpy as np

def guesses(replicatsCRTG, replicatsCRT2617, n_genes, n_replicats, times) :
    # prior for etas
    replicatsCRT2617_normalized = replicatsCRT2617[:, :, 1:] / replicatsCRT2617[:, :, 0, np.newaxis]
    eta0 = np.nanmean(replicatsCRT2617_normalized, axis = (1, ))
    
    # prior for alphas
    alpha0 = 0.2 * np.ones(n_genes)
    
    # prior for betas
    beta0 = np.nanmean(replicatsCRTG[:, : , 1, :], axis = (0, 1))
    
    # prior for gammas
    gamma0 = eta0.max() * np.ones(n_genes)
    
    # prior for deltas
    delta0 = np.zeros(n_genes)
    
    for g in range(n_genes): # g goes from 0 to n_genes-1
        xdata = times[3:9] - times[3]
        ydataRT = replicatsCRTG[0, :, 3:9, g]
        ydataRT[2, 1] = np.nanmean(ydataRT[ :, 1]) # on se prémunit d'un NAN en faisant la moyenne sur les réplicats
#        print('ydata.shape', ydata.shape)
#        print('ydata', ydata)
        popt = np.zeros(7)
        for r in range(n_replicats): # r goes from 0 to n_replicats-1
            g3 = replicatsCRTG[0, r, 3, g]
            def expo(t, delta):
                return g3 * np.exp(-delta * t)
            popt[r], pcov = curve_fit(expo, xdata, ydataRT[r])
        delta0[g] = np.nanmean(popt)

    return alpha0, beta0, gamma0, delta0, eta0
