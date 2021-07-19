'''
Created on Feb 23, 2021

@author: Miguel
'''

class Enum(object):
    @classmethod
    def members(cls):
        import inspect
        result = []
        for i in inspect.getmembers(cls):
            name = i[0]
            value = i[1]
            if not (name.startswith('_') or inspect.ismethod(value)):
                result.append(value)
        return result

#===============================================================================
#  DEFINED ENUMERATIONS
#===============================================================================

class InputParts(Enum):
    Interaction_Title = 'Interaction_Title'
    Output_Parameters = 'Output_Parameters'
    SHO_Parameters = 'SHO_Parameters'
    Valence_Space = 'Valence_Space'
    Core = 'Core'
    Force_Parameters = 'Force_Parameters'

class Output_Parameters(Enum):
    Output_Filename = 'Output_Filename'
    Hamil_Type      = 'Hamil_Type'
    COM_correction  = 'COM_correction'

class SHO_Parameters(Enum):
    A_Mass      = 'A_mass'
    hbar_omega  = 'hbar_omega'
    b_length    = 'b_length'

class ValenceSpaceParameters(Enum):
    Q_Number   = 'Q_Number'
    QN_Energies = 'QN_Energies'
    l_great_than_10 = 'l_great_than_10'

class ForceParameters(Enum):
    SDI = 'SDI'
    Central = 'Central'
    Coulomb = 'Coulomb'
    Tensor  = 'Tensor'
    SpinOrbit = 'SpinOrbit'
    SpinOrbitShortRange = 'SpinOrbitShortRange'
    Brink_Boeker  = 'Brink_Boeker'
    Density_Dependent = 'Density_Dependent'
    Kinetic_2Body = 'Kinetic_2Body'
    Multipole_Expansion = 'Multipole_Expansion'


class PotentialForms(Enum):
    Gaussian    = 'gaussian'                # exp(-(r/mu_)^2)
    Exponential = 'exponential'             # exp(-r/mu_)
    Coulomb     = 'coulomb'                 # mu_/r
    Yukawa      = 'yukawa'                  # exp(-r/mu_) / (r/mu_)
    Power       = 'power'                   # (r/mu_)^n_power
    Gaussian_power = 'gaussian_power'       # exp(-(r/mu_)^2) / (r/mu_)^n_power

#===============================================================================
# FORCE PARAMETERS DEFINITIONS
#===============================================================================

class CentralMEParameters(Enum):
    potential = 'potential'
    constant  = 'constant'
    mu_length = 'mu_length'
    n_power   = 'n_power'

class BrinkBoekerParameters(Enum):
    mu_length   = 'mu_length'
    Wigner      = 'Wigner'
    Majorana    = 'Majorana'
    Bartlett    = 'Bartlett'
    Heisenberg  = 'Heisenberg'

class DensityDependentParameters(Enum):
    constant    = 'constant'
    x0    = 'x0'
    alpha = 'alpha'

# TODO: Update when adding forces, Enum implementations must be given
# TODO: Implement also the attribute names in AttributeArgs.ForceArgs
## Use Enum if the interaction has no parameters

ForceVariablesDict = {
    ForceParameters.Brink_Boeker : BrinkBoekerParameters,
    ForceParameters.Central : CentralMEParameters,
    ForceParameters.Coulomb : Enum,
    ForceParameters.Tensor  : CentralMEParameters,
    ForceParameters.SpinOrbit : CentralMEParameters,
    ForceParameters.SpinOrbitShortRange : CentralMEParameters,
    ForceParameters.Density_Dependent : DensityDependentParameters,
    ForceParameters.Kinetic_2Body : Enum,
    ForceParameters.SDI : None,
    ForceParameters.Multipole_Expansion: None,
}

# from the input name for the force to a
## TODO: this dictionary is deprecated (Remove)
# ForceVariablesMatrixElementDict = {
#     ForceParameters.Brink_Boeker : 'BrinkBoeker',
#     ForceParameters.Central      : 'CentralForce',
#     ForceParameters.Coulomb      : 'Coulomb',
#     ForceParameters.Tensor       : 'TensorForce',
#     ForceParameters.SpinOrbit    : 'SpinOrbitForce',
#     ForceParameters.SpinOrbitShortRange : 'SpinOrbitShortRange',
#     ForceParameters.Density_Dependent   : ''     
# }

class AttributeArgs(Enum):
    name    = 'name'
    details = 'details'
    value   = 'value'
    units   = 'units'
    
    class ValenceSpaceArgs(Enum):
        sp_state  = 'sp_state'
        sp_energy = 'sp_energy'
    
    class CoreArgs(Enum):
        protons     = 'protons'
        neutrons    = 'neutrons'
        innert_core = 'innert_core'
    
    class ForceArgs(Enum):
        active = 'active'
        
        class Brink_Boeker(Enum):
            part_1 = 'part_1'
            part_2 = 'part_2'
        
        class Potential(Enum):
            name = 'name'

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



