from helpers.Enums import ForceEnum

from .BrinkBoeker import BrinkBoeker, GaussianSeries_JTScheme
from matrix_elements.TensorForces import TensorForce, TensorForce_JTScheme
from matrix_elements.CentralForces import CentralForce, CentralForce_JTScheme,\
    CoulombForce, DensityDependentForce_JTScheme, KineticTwoBody_JTScheme
from matrix_elements.SpinOrbitForces import SpinOrbitForce, SpinOrbitForce_JTScheme, \
    ShortRangeSpinOrbit_JTScheme


def switchMatrixElementType(force, J_scheme=False):
    """
    Select defined force classes for implementing matrix elements.
    Call from TBME_Runner automatic selection
    
    :force  From ForceEnum(Enum)
    TODO: :scheme (J and JT)
    """
    
    if force == ForceEnum.Brink_Boeker:
        return BrinkBoeker
    elif force == ForceEnum.GaussianSeries:
        return GaussianSeries_JTScheme
    elif force == ForceEnum.Central:
        return CentralForce_JTScheme
    elif force == ForceEnum.Coulomb:
        return CoulombForce
    elif force == ForceEnum.Kinetic_2Body:
        return KineticTwoBody_JTScheme
    elif force == ForceEnum.Density_Dependent:
        return DensityDependentForce_JTScheme
    
    elif force == ForceEnum.SpinOrbit:
        return SpinOrbitForce_JTScheme
    elif force == ForceEnum.SpinOrbitShortRange:
        return ShortRangeSpinOrbit_JTScheme
    
    elif force == ForceEnum.Tensor:
        return TensorForce_JTScheme
    else:
        raise Exception("Invalid force: [{}]".format(force))