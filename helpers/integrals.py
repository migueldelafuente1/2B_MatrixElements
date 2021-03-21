'''
Created on Mar 12, 2021

@author: Miguel
'''
import numpy as np
from helpers.Enums import PotentialForms
from helpers.Helpers import gamma_half_int, fact

def talmiIntegral(p, potential, b_param, mu_param, n_power=0):
    """
    :p          index order
    :potential  form of the potential, from Poten
    :b_param    SHO length parameter
    :mu_param   force proportional coefficient (by interaction definition)
    :n_power    auxiliary parameter for power dependent potentials
    """
    # TODO: Might implement potential args by Enum and select them in the switch
    
    if potential == PotentialForms.Gaussian:
        return b_param / (1 + (b_param/mu_param)**2)**(p+1.5)
    
    elif potential == PotentialForms.Coulomb:
        # mu_param_param must be fixed for the nucleus (Z)
        return mu_param * (b_param**2) * np.exp(gamma_half_int(p + 1.5))
    
    elif potential == PotentialForms.Yukawa:
        sum_ = 0.
        for i in range(2*p+1 +1):
            aux = fact(2*p + 1) - fact(i) - fact(2*p + 1 - i)
            if i % 2 == 0:
                aux += fact((i + 1)//2)
            else:
                aux += gamma_half_int(i + 1)
            
            aux += (2*p + 1 - i) * np.log(b_param/(2*mu_param))
            
            sum_ += (-1)**(2*p + 1 - i) * np.exp(aux)
            
        sum_ *= mu_param * (b_param**2) * np.exp((b_param/(2*mu_param))**2) 
        sum_ /= np.exp(gamma_half_int(p + 1.5))
        
        return sum_
    
    elif potential == PotentialForms.Power:
        if n_power == 0:
            return b_param**3
        aux = gamma_half_int(2*p + 3 + n_power) - gamma_half_int(2*p + 3)
        
        return np.exp(aux + n_power*(3*np.log(b_param) - np.log(mu_param)))
        
    elif potential == PotentialForms.Gaussian_power:
        raise Exception("Talmi integral 'gaussian_power' not implemented")
    else:
        raise Exception("Talmi integral [{}] is not defined, valid potentials: {}"
                        .format(potential, PotentialForms.members()))
