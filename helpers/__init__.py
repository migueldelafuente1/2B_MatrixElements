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
    import sympy
    import numpy
except ImportError as err:
    print("FATAL ERROR :: numpy and sympy are required to run the program. STOP.")
    raise err

try:
    import scipy
except ImportError as err:
    SCIPY_INSTALLED = False
    print("WARNING :: "+str(err)+".\n It cannot be evaluated Central for Yukawa/Exponential")
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

## Exception Message
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# ImportError: No module named 'pandas'
