'''
Created on Feb 23, 2021

@author: Miguel
'''

class Enum(object):
    @classmethod
    def members(cls, but=[]):
        """
        :but excludes attributes if found
        """
        import inspect
        result = []
        for i in inspect.getmembers(cls):
            name = i[0]
            value = i[1]
            if not (name.startswith('_') or inspect.ismethod(value)):
                result.append(value)
        
        if but:
            for value in but:
                if value in result:
                    result.remove(value)
                
        return result
        
#===============================================================================
#  DEFINED ENUMERATIONS 
#===============================================================================

class InputParts(Enum):
    Interaction_Title = 'Interaction_Title'
    Output_Parameters = 'Output_Parameters'
    SHO_Parameters = 'SHO_Parameters'
    Valence_Space  = 'Valence_Space'
    Core = 'Core'
    Force_Parameters = 'Force_Parameters'

class Output_Parameters(Enum):
    Output_Filename = 'Output_Filename'
    Hamil_Type      = 'Hamil_Type'
    COM_correction  = 'COM_correction'

class SHO_Parameters(Enum):
    A_Mass      = 'A_mass'
    Z           = 'Z'
    hbar_omega  = 'hbar_omega'
    b_length    = 'b_length'

class ValenceSpaceParameters(Enum):
    Q_Number   = 'Q_Number'
    QN_Energies = 'QN_Energies'
    l_great_than_10 = 'l_great_than_10'

class CoreParameters(Enum):
    protons     = 'protons'
    neutrons    = 'neutrons'
    innert_core = 'innert_core'
    energy      = 'energy' 
        
class ForceEnum(Enum):
    SDI = 'SDI'
    Delta   = 'Delta'
    Central = 'Central'
    Coulomb = 'Coulomb'
    Tensor  = 'Tensor'
    SpinOrbit = 'SpinOrbit'
    SpinOrbitFiniteRange = 'SpinOrbitFiniteRange'
    SpinOrbitShortRange  = 'SpinOrbitShortRange'
    Quadratic_SpinOrbit  = 'Quadratic_SpinOrbit'
    Brink_Boeker    = 'Brink_Boeker'
    PotentialSeries = 'PotentialSeries'
    YukawiansM3Y    = 'M3Y_yukawians'
    M3YTensor       = 'M3Y_tensor'
    M3YSpinOrbit    = 'M3Y_SpinOrbit'
    Density_Dependent    = 'Density_Dependent'
    Density_Dependent_From_File = 'Density_From_File'
    Density_FiniteRange  = 'DensityFiniteRange'
    Quadratic_OrbitalMomentum = 'Quadratic_OrbitalMomentum'
    Kinetic_2Body    = 'Kinetic_2Body'
    Kinetic_Total    = 'Kinetic_Total'
    Multipole_Delta  = 'Multipole_Delta'
    Multipole_Moment = 'Multipole_Moment'
    SkyrmeBulk = 'SkyrmeBulk'
    TensorS12  = 'TensorS12'
    Force_From_File = 'Force_From_File'
    CentralGeneralized   = 'CentralGeneralized'
    Argone14NuclearTerms = 'Argone14NuclearTerms'
    Argone18NuclearTerms = 'Argone18NuclearTerms'
    Argone18Electromagetic = 'Argone18Electromagetic'

class PotentialForms(Enum):
    Gaussian    = 'gaussian'                # exp(-(r/mu_)^2)
    Exponential = 'exponential'             # exp(-r/mu_)
    Coulomb     = 'coulomb'                 # mu_/r
    Delta       = 'delta'                   # delta(r)/r^2
    Yukawa      = 'yukawa'                  # exp(-r/mu_) / (r/mu_)
    Power       = 'power'                   # (r/mu_)^n_power
    Gaussian_power = 'gaussian_power'       # exp(-(r/mu_)^2) * (r/mu_)^n_power
    Wood_Saxon  = 'wood_saxon'              # (r/mu_)^n_power /( 1 + exp((r-mu_2 * A^1/3)/mu_3) )
    Exponential_power = 'exponential_power' # exp(-r/mu_) * (r/mu_)^n_power
    YukawaGauss_power = 'gauss_yukawa_power'# exp(-(r/mu_)-(r/mu_2)^2) * (r/mu_)^n_power

#===============================================================================
# FORCE PARAMETERS DEFINITIONS
#===============================================================================

class CentralMEParameters(Enum):
    potential = 'potential'
    constant  = 'constant'
    mu_length = 'mu_length'
    n_power   = 'n_power'
    opt_mu_2  = 'opt_mu_2'   # optional
    opt_mu_3  = 'opt_mu_3'   # optional
    opt_cutoff= 'opt_cutoff' # optional (1 - exp(-(r/cutoff_len)^2))

class BrinkBoekerParameters(Enum):
    mu_length   = 'mu_length'
    Wigner      = 'Wigner'
    Majorana    = 'Majorana'
    Bartlett    = 'Bartlett'
    Heisenberg  = 'Heisenberg'

class CentralWithExchangeParameters(CentralMEParameters, BrinkBoekerParameters):
    pass

# class CentralWithExchangeParameters(Enum):
#     potential = 'potential'
#     constant  = 'constant'
#     mu_length = 'mu_length'
#     n_power   = 'n_power'
#     Wigner    = 'Wigner'
#     Majorana  = 'Majorana'
#     Bartlett  = 'Bartlett'
#     Heisenberg= 'Heisenberg'

class CentralGeneralizedMEParameters(CentralMEParameters):
    potential_R = 'potential_R'
    constant_R  = 'constant_R'
    mu_length_R = 'mu_length_R'
    n_power_R   = 'n_power_R'

class PotentialSeriesParameters(Enum):
    part    = 'part'

class DensityDependentParameters(Enum):
    constant    = 'constant' # t_0
    x0    = 'x0'
    alpha = 'alpha'
    core  = 'core'
    file  = 'file'
    integration = 'integration'
    
class DensityAndExchangeParameters(CentralWithExchangeParameters, 
                                   DensityDependentParameters):
    pass
    
class SkyrmeBulkParameters(Enum):
    t0 = 't0'
    t1 = 't1'
    t2 = 't2'
    x0 = 'x0'

class SDIParameters(Enum):
    constants = 'constants'

class MultipoleParameters(Enum):
    constants = 'constants'

class ForceFromFileParameters(Enum):
    file  = 'file'
    options = 'options'
    #scheme  = 'scheme' # optional, not implemented

# TODO: Update when adding forces, Enum implementations must be given
# TODO: Implement also the attribute names in AttributeArgs.ForceArgs
## Use Enum if the interaction has no parameters

ForceVariablesDict = {
    ForceEnum.Brink_Boeker    : BrinkBoekerParameters,
    ForceEnum.PotentialSeries : PotentialSeriesParameters,
    ForceEnum.YukawiansM3Y    : BrinkBoekerParameters,
    ForceEnum.M3YTensor       : BrinkBoekerParameters,
    ForceEnum.M3YSpinOrbit    : BrinkBoekerParameters,
    ForceEnum.Central   : CentralWithExchangeParameters,    # CentralMEParameters DEP
    ForceEnum.Coulomb   : Enum,
    ForceEnum.Tensor    : CentralMEParameters,
    ForceEnum.TensorS12 : CentralWithExchangeParameters,
    ForceEnum.SpinOrbit : CentralMEParameters,
    ForceEnum.SpinOrbitShortRange : CentralMEParameters,
    ForceEnum.SpinOrbitFiniteRange: CentralWithExchangeParameters,
    ForceEnum.Quadratic_SpinOrbit : CentralWithExchangeParameters,
    ForceEnum.Density_Dependent   : DensityDependentParameters,
    ForceEnum.Density_Dependent_From_File: DensityDependentParameters,
    ForceEnum.Density_FiniteRange        : DensityAndExchangeParameters, 
    ForceEnum.Quadratic_OrbitalMomentum  : CentralWithExchangeParameters,
    ForceEnum.SkyrmeBulk    : SkyrmeBulkParameters, 
    ForceEnum.Kinetic_2Body : Enum,
    ForceEnum.Kinetic_Total : Enum,
    ForceEnum.SDI           : SDIParameters,
    ForceEnum.Delta         : BrinkBoekerParameters,
    ForceEnum.Multipole_Delta : MultipoleParameters,
    ForceEnum.Multipole_Moment: CentralMEParameters, 
    ForceEnum.Force_From_File : ForceFromFileParameters,
    ForceEnum.CentralGeneralized : CentralGeneralizedMEParameters,
    ForceEnum.Argone18NuclearTerms : Enum,
    ForceEnum.Argone14NuclearTerms : Enum,
    ForceEnum.Argone18Electromagetic: Enum,
}

ForcesWithRepeatedParametersList = [
    ForceEnum.PotentialSeries,
]
class AttributeArgs(Enum):
    name    = 'name'
    details = 'details'
    value   = 'value'
    units   = 'units'
        
    class ValenceSpaceArgs(Enum):
        sp_state  = 'sp_state'
        sp_energy = 'sp_energy'
    
    class CoreArgs(Enum):
        name        = 'name'
        protons     = 'protons'
        neutrons    = 'neutrons'
        
    class ForceArgs(Enum):
        active = 'active'
        
        class Brink_Boeker(Enum):
            part_1 = 'part_1'
            part_2 = 'part_2'
        
        class Potential(Enum):
            name = 'name'
        
        class SDI(Enum):
            A_T0 = 'AT0'
            A_T1 = 'AT1'
            B = 'B' # when activated, MSDI interaction
            C = 'C'
            # Brussaard_ & Glaudemans_ book (1977)
        class Multipole(Enum):
            A = 'A'  # central
            B = 'B'  # spin
            C = 'C'  # isospin
            D = 'D'  # spin-isospin
            ## Suhonen expression for spin - isospin multipole interaction
        class DensDep(Enum):
            protons  = 'protons'
            neutrons = 'neutrons'
            core_b_len = 'core_b_len'
            r_dim      = 'r_dim'
            omega_ord  = 'omega_ord'
        
    class FileReader(Enum):
        ignorelines = 'ignorelines'
        constant    = 'constant'
        l_ge_10     = 'l_ge_10'

class Units(Enum):
    MeV = 'MeV'
    fm  = 'fm'
    
class CouplingSchemeEnum(Enum):
    L   = 'Total Orbital Angular Momentum'
    S   = 'Total Spin'
    JJ  = 'Total Angular Momentum'
    T   = 'Total Isospin'

# TODO: Method to implement a folder for the results
OUTPUT_FOLDER = 'results'

class OutputFileTypes(Enum):
    sho     = '.sho'
    oneBody = '.01b'
    twoBody = '.2b'
    centerOfMass = '.com'



