from helpers.Enums import ForceEnum

from .BrinkBoeker import BrinkBoeker, PotentialSeries_JTScheme
from .TensorForces import TensorForce, TensorForce_JTScheme, TensorS12_JTScheme
from .CentralForces import CentralForce, CentralForce_JTScheme,\
    CoulombForce, KineticTwoBody_JTScheme, Quadratic_OrbitalMomentum_JTScheme,\
    CentralGeneralizedForce_JTScheme
from .DensityForces import DensityDependentForceFromFile_JScheme, \
    DensityDependentForce_JTScheme, DensityFiniteRange_JTScheme
    
from .SpinOrbitForces import SpinOrbitForce,\
    SpinOrbitForce_JTScheme, ShortRangeSpinOrbit_JTScheme, \
    SpinOrbitFiniteRange_JTScheme, Quadratic_SpinOrbit_JTScheme
from .MatrixElement import _MatrixElementReader
from .ZeroRangeForces import SDI_JTScheme, Delta_JTScheme
from .SkyrmeForces import SkrymeBulk_JTScheme
from .MultipoleForces import MultipoleDelta_JTScheme, MultipoleMoment_JTScheme
from .BrinkBoeker import YukawiansM3Y_JTScheme, YukawiansM3Y_tensor_JTScheme, \
    YukawiansM3Y_SpinOrbit_JTScheme
from matrix_elements.ArgonePotential import NucleonAv18TermsInteraction_JTScheme,\
    NucleonAv14TermsInteraction_JTScheme, ElectromagneticAv18TermsInteraction_JScheme
from matrix_elements.SkyrmeForces import ShortRangeSpinOrbit_COM_JTScheme
from matrix_elements.MomentumForces import RelativeMomentumSquared_JTScheme,\
    TotalMomentumSquared_JTScheme


def switchMatrixElementType(force, J_scheme=False):
    """
    Select defined force classes for implementing matrix elements.
    Call from TBME_Runner automatic selection
    
    :force  From ForceEnum(Enum)
    TODO: :scheme (J and JT) ??
    """
    ## CENTRAL - EXCHANGED - SERIES
    
    if force == ForceEnum.Brink_Boeker:
        return BrinkBoeker
    elif force == ForceEnum.PotentialSeries:
        return PotentialSeries_JTScheme
    elif force == ForceEnum.YukawiansM3Y:
        return YukawiansM3Y_JTScheme
    elif force == ForceEnum.M3YTensor:
        return YukawiansM3Y_tensor_JTScheme
    elif force == ForceEnum.M3YSpinOrbit:
        return YukawiansM3Y_SpinOrbit_JTScheme
    
    ## CENTRALS: 
    
    elif force == ForceEnum.Central:
        return CentralForce_JTScheme
    elif force == ForceEnum.Coulomb:
        return CoulombForce
    elif force == ForceEnum.Kinetic_2Body:
        return KineticTwoBody_JTScheme
    elif force == ForceEnum.Kinetic_Total:
        # return RelativeMomentumSquared_JTScheme
        return TotalMomentumSquared_JTScheme
    elif force == ForceEnum.Quadratic_OrbitalMomentum:
        return Quadratic_OrbitalMomentum_JTScheme
    elif force == ForceEnum.Density_Dependent:
        return DensityDependentForce_JTScheme
    elif force == ForceEnum.Density_Dependent_From_File:
        return DensityDependentForceFromFile_JScheme
    elif force == ForceEnum.Density_FiniteRange:
        return DensityFiniteRange_JTScheme
    elif force == ForceEnum.SDI:
        return SDI_JTScheme
    elif force == ForceEnum.Delta:
        return Delta_JTScheme
    elif force == ForceEnum.Multipole_Delta:
        return MultipoleDelta_JTScheme
    elif force == ForceEnum.Multipole_Moment:
        return MultipoleMoment_JTScheme
    elif force == ForceEnum.CentralGeneralized:
        return CentralGeneralizedForce_JTScheme
    
    ## SPIN-ORBIT
    
    elif force == ForceEnum.SpinOrbit:
        return SpinOrbitForce_JTScheme
    elif force == ForceEnum.SpinOrbitShortRange:
        # return ShortRangeSpinOrbit_COM_JTScheme
        return ShortRangeSpinOrbit_JTScheme
    elif force == ForceEnum.SpinOrbitFiniteRange:
        return SpinOrbitFiniteRange_JTScheme
    elif force == ForceEnum.Quadratic_SpinOrbit:
        return Quadratic_SpinOrbit_JTScheme
    elif force == ForceEnum.SkyrmeBulk:
        return SkrymeBulk_JTScheme
    
    ## TENSOR
    
    elif force == ForceEnum.Tensor:
        return TensorForce_JTScheme
    elif force == ForceEnum.TensorS12:
        return TensorS12_JTScheme
    
    ## OTHERS
    
    elif force == ForceEnum.Argone14NuclearTerms:
        return NucleonAv14TermsInteraction_JTScheme
    elif force == ForceEnum.Argone18NuclearTerms:
        return NucleonAv18TermsInteraction_JTScheme
    elif force == ForceEnum.Argone18Electromagetic:
        return ElectromagneticAv18TermsInteraction_JScheme
    
    if force == ForceEnum.Force_From_File:
        return _MatrixElementReader
    else:
        raise Exception("Invalid force: [{}]".format(force))