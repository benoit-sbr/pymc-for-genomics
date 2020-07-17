# Chargement des bibliothèques
import numpy as np

def expression(t, temps, mu0, alpha, beta, gamma, delta, eta):
    # Docstring de la fonction
    """
    Renvoie l'expression de mu (équation juste après (9) dans le papier Khanin et al)
    Positional arguments:
    t		--- temps où l'on calcule l'expression
    temps	--- vecteur de subdivision du temps (instants de mesure)
    mu0		--- niveau d'expression à t = 0
    alpha	---
    beta	---
    gamma	---
    delta	---
    eta		--- vecteur des différents valeurs de eta dans la subdivision
    Keyword arguments:
    """
    # Corps de la fonction
    tau = np.linspace(0., t, t.size)
#    print('tau.shape', tau.shape)
    positions = np.searchsorted(temps, tau, side='right') - 1
#    print('positions.shape', positions.shape)
#    print('eta[positions].shape', eta[positions].shape)
    integrand = np.exp(delta*tau) * eta[positions] / (gamma + eta[positions])
    integrale = np.trapz(integrand, tau)
    half1st = (mu0 - alpha/delta) * np.exp(-delta*t) + alpha/delta
#    print (beta.shape, (delta*t).shape, integrale.shape)
    half2nd = beta * np.exp(-delta*t) * integrale
    result  = half1st + half2nd
    return half1st, half2nd, result, eta[positions]
