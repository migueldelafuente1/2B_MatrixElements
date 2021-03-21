from helpers.Enums import ForceParameters

from .BrinkBoeker import BrinkBoeker
from matrix_elements.TensorForces import TensorForce, TensorForce_JTScheme
from matrix_elements.CentralForces import CentralForce, CentralForce_JTScheme
from matrix_elements.SpinOrbitForces import SpinOrbitForce, SpinOrbitForce_JTScheme


def switchMatrixElementType(force):
    """
    Select defined force classes for implementing matrix elements.
    Call from TBME_Runner automatic selection
    
    :force  From ForceParameters(Enum)
    """
    
    if force == ForceParameters.Brink_Boeker:
        return BrinkBoeker
    elif force == ForceParameters.Central:
        return CentralForce_JTScheme
    elif force == ForceParameters.Spin_Orbit:
        return SpinOrbitForce_JTScheme
    elif force == ForceParameters.Tensor:
        return TensorForce_JTScheme
    else:
        raise Exception("Invalid force: [{}]".format(force))