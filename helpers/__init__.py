'''
Created on Oct 8, 2021

@author: Miguel
'''
#===============================================================================
# %% Importation of required libraries
#===============================================================================

SCIPY_INSTALLED  = True
PANDAS_INSTALLED = True
MATPLOTLIB_INSTALLED   = True
## SYMPY is required to evaluate Angular coeffs_, NUMPY is fundamental (raise error)

try:
    import scipy
except ImportError as err:
    SCIPY_INSTALLED = False
    print("WARNING :: "+str(err)+". (i.e) cannot evaluate Central for Yukawa")
try:
    import pandas
except ImportError as err:
    PANDAS_INSTALLED = False
    print("WARNING :: "+str(err)+" (i.e) cannot evaluate certain tests")
try:
    import matplotlib
except ImportError as err:
    MATPLOTLIB_INSTALLED = False
    print("WARNING :: "+str(err)+" (i.e) Do not evaluate test modules.")
