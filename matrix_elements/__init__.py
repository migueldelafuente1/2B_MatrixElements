from helpers.Enums import ForceEnum

from .BrinkBoeker import BrinkBoeker, PotentialSeries_JTScheme
from matrix_elements.TensorForces import TensorForce, TensorForce_JTScheme,\
    TensorS12_JTScheme
from matrix_elements.CentralForces import CentralForce, CentralForce_JTScheme,\
    CoulombForce, DensityDependentForce_JTScheme, KineticTwoBody_JTScheme,\
    MultipoleDelta_JTScheme
from matrix_elements.SpinOrbitForces import SpinOrbitForce, SpinOrbitForce_JTScheme, \
    ShortRangeSpinOrbit_JTScheme
from matrix_elements.MatrixElement import _MatrixElementReader
from matrix_elements.ZeroRangeForces import SDI_JTScheme
from matrix_elements.SkyrmeForces import SkrymeBulk_JTScheme


def switchMatrixElementType(force, J_scheme=False):
    """
    Select defined force classes for implementing matrix elements.
    Call from TBME_Runner automatic selection
    
    :force  From ForceEnum(Enum)
    TODO: :scheme (J and JT) ??
    """
    
    if force == ForceEnum.Brink_Boeker:
        return BrinkBoeker
    elif force == ForceEnum.PotentialSeries:
        return PotentialSeries_JTScheme
    elif force == ForceEnum.Central:
        return CentralForce_JTScheme
    elif force == ForceEnum.Coulomb:
        return CoulombForce
    elif force == ForceEnum.Kinetic_2Body:
        return KineticTwoBody_JTScheme
    elif force == ForceEnum.Density_Dependent:
        return DensityDependentForce_JTScheme
    elif force == ForceEnum.SDI:
        return SDI_JTScheme
    elif force == ForceEnum.Multipole_Delta:
        return MultipoleDelta_JTScheme
    
    elif force == ForceEnum.SpinOrbit:
        return SpinOrbitForce_JTScheme
    elif force == ForceEnum.SpinOrbitShortRange:
        return ShortRangeSpinOrbit_JTScheme
    elif force == ForceEnum.SkyrmeBulk:
        return SkrymeBulk_JTScheme
    
    elif force == ForceEnum.Tensor:
        return TensorForce_JTScheme
    elif force == ForceEnum.TensorS12:
        return TensorS12_JTScheme
    
    if force == ForceEnum.Force_From_File:
        return _MatrixElementReader
    else:
        raise Exception("Invalid force: [{}]".format(force))