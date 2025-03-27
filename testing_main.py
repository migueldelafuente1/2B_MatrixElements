'''
Created on 3 dic 2024

@author: delafuente
'''
from helpers.Enums import CentralGeneralizedMEParameters, SHO_Parameters, BrinkBoekerParameters,\
    PotentialForms, CentralMEParameters, PotentialSeriesParameters,\
    DensityDependentParameters
from matrix_elements.CentralForces import CentralGeneralizedForce_JTScheme
from matrix_elements.CentralForces import CentralForce_JTScheme
from helpers.WaveFunctions import QN_2body_jj_JT_Coupling, QN_1body_jj
from matrix_elements.BrinkBoeker import PotentialSeries_JTScheme
from matrix_elements.ZeroRangeForces import Delta_JTScheme
from matrix_elements.DensityForces import DensityDependentForce_JTScheme
from matrix_elements.TensorForces import TensorS12_JTScheme
from matrix_elements.ArgonePotential import NucleonAv18TermsInteraction_JTScheme,\
    NucleonAv14TermsInteraction_JTScheme

if __name__ == '__main__':
    
    " 0 5 001 001 101 101 0 1"
    J = 0
    T = 1
    bra = QN_2body_jj_JT_Coupling(QN_1body_jj(0,2,5, mt= 1),
                                  QN_1body_jj(0,2,5, mt=-1), J, T)
    ket = QN_2body_jj_JT_Coupling(QN_1body_jj(0,2,5, mt= 1),
                                  QN_1body_jj(0,2,5, mt=-1), J, T)
    # ket = QN_2body_jj_JT_Coupling(QN_1body_jj(0,1,1, mt=-1),
    #                               QN_1body_jj(0,0,1, mt=-1), J, T)
    # bra = QN_2body_jj_JT_Coupling(QN_1body_jj(5,1,1, mt=-1),
    #                               QN_1body_jj(5,1,1, mt=-1), J, T)
    # ket = QN_2body_jj_JT_Coupling(QN_1body_jj(5,1,1, mt=-1),
    #                               QN_1body_jj(5,1,1, mt=-1), J, T)
    
    kwargs = {
        CentralMEParameters.potential: {'name': PotentialForms.Gaussian},
        # CentralMEParameters.opt_mu_2 : {'value': 0.2,},
        # CentralMEParameters.opt_mu_3 : {'value': 0.5,},
        # CentralMEParameters.n_power  : {'value': 0,},
        CentralMEParameters.mu_length: {'value': 1.4295301308423947,},
        BrinkBoekerParameters.Wigner:     {'value': 1.0,} ,
        # BrinkBoekerParameters.Bartlett:   {'value': 0,},
        # BrinkBoekerParameters.Heisenberg: {'value': 0,},
        # BrinkBoekerParameters.Majorana:   {'value': 0,},
        SHO_Parameters.b_length: 1.500,
    }
    Inter_ = NucleonAv14TermsInteraction_JTScheme
    Inter_._SWITCH_OFF_CONSTANTS = True
    for term in Inter_._SWITCH_OFF_TERMS.keys():
        if term == Inter_.TermEnum.t: continue
        Inter_._SWITCH_OFF_TERMS[term] = True
    
    NucleonAv14TermsInteraction_JTScheme.turnDebugMode(True)
    NucleonAv14TermsInteraction_JTScheme.setInteractionParameters(**kwargs)
    me = NucleonAv14TermsInteraction_JTScheme(ket, bra)
    print(me.value)
    me.saveXLog('tensor-av14')
    0/0
    
    # kwargs = {
    #     CentralGeneralizedMEParameters.potential:   {'name': 'gaussian'},
    #     CentralGeneralizedMEParameters.constant :   {'value': 1,},
    #     CentralGeneralizedMEParameters.mu_length:   {'value': 10000,},
    #     CentralGeneralizedMEParameters.constant_R:  {'value': 1,},
    #     CentralGeneralizedMEParameters.mu_length_R: {'value': 1,},
    #     CentralGeneralizedMEParameters.potential_R: {'name': 'gaussian'},
    #     BrinkBoekerParameters.Wigner:     {'value': 1000,} ,
    #     BrinkBoekerParameters.Bartlett:   {'value': 0,},
    #     BrinkBoekerParameters.Heisenberg: {'value': 0,},
    #     BrinkBoekerParameters.Majorana:   {'value': 0,},
    #     SHO_Parameters.b_length: 1.8,
    # }
    # CentralGeneralizedForce_JTScheme.turnDebugMode(True)
    # CentralGeneralizedForce_JTScheme.setInteractionParameters(**kwargs)
    # me = CentralGeneralizedForce_JTScheme(ket, bra)
    # print(me.value)
    # me.saveXLog('FR_gauss_gR')
    mu = 0.75
    b_len = 1.8
    kwargs = {
        'part 1': {'potential':'gaussian_power', 'constant':'    0.202', 'mu_length':'1.365423', 'n_power':'-1'},
        'part 2': {'potential':'gaussian_power', 'constant':'    0.175', 'mu_length':'0.910282', 'n_power':'-1'},
        'part 3': {'potential':'gaussian_power', 'constant':'    0.169', 'mu_length':'0.606855', 'n_power':'-1'},
        'part 4': {'potential':'gaussian_power', 'constant':'    0.170', 'mu_length':'0.404570', 'n_power':'-1'},
        'part 5': {'potential':'gaussian_power', 'constant':'    0.173', 'mu_length':'0.269713', 'n_power':'-1'},
        'part 6': {'potential':'gaussian_power', 'constant':'    0.176', 'mu_length':'0.179809', 'n_power':'-1'},
        'part 7': {'potential':'gaussian_power', 'constant':'    0.178', 'mu_length':'0.119873', 'n_power':'-1'},
        'part 8': {'potential':'gaussian_power', 'constant':'    0.179', 'mu_length':'0.079915', 'n_power':'-1'},
        'part 9': {'potential':'gaussian_power', 'constant':'    0.181', 'mu_length':'0.053277', 'n_power':'-1'},
        'part 10': {'potential':'gaussian_power', 'constant':'    0.181', 'mu_length':'0.035518', 'n_power':'-1'},
        'part 11': {'potential':'gaussian_power', 'constant':'    0.182', 'mu_length':'0.023679', 'n_power':'-1'},
        'part 12': {'potential':'gaussian_power', 'constant':'    0.182', 'mu_length':'0.015786', 'n_power':'-1'},
        'part 13': {'potential':'gaussian_power', 'constant':'    0.183', 'mu_length':'0.010524', 'n_power':'-1'},
        'part 14': {'potential':'gaussian_power', 'constant':'    0.183', 'mu_length':'0.007016', 'n_power':'-1'},
        'part 15': {'potential':'gaussian_power', 'constant':'    0.183', 'mu_length':'0.004677', 'n_power':'-1'},
        'part 16': {'potential':'gaussian_power', 'constant':'    0.183', 'mu_length':'0.003118', 'n_power':'-1'},
        'part 17': {'potential':'gaussian_power', 'constant':'    0.183', 'mu_length':'0.002079', 'n_power':'-1'},
        'part 18': {'potential':'gaussian_power', 'constant':'    0.183', 'mu_length':'0.001386', 'n_power':'-1'},
        'part 19': {'potential':'gaussian_power', 'constant':'    0.183', 'mu_length':'0.000924', 'n_power':'-1'},
        'part 20': {'potential':'gaussian_power', 'constant':'    0.183', 'mu_length':'0.000616', 'n_power':'-1'},
        SHO_Parameters.b_length: b_len,
        }
    PotentialSeries_JTScheme.turnDebugMode(False)
    PotentialSeries_JTScheme.setInteractionParameters(**kwargs)
    me = PotentialSeries_JTScheme(ket, bra)
    print(me.value)
    
    kwargs = {
        BrinkBoekerParameters.Wigner:     {'value': 1,} ,
        BrinkBoekerParameters.Bartlett:   {'value': 0,},
        BrinkBoekerParameters.Heisenberg: {'value': 0,},
        BrinkBoekerParameters.Majorana:   {'value': 0,},
        CentralMEParameters.potential:    {'name': PotentialForms.Yukawa},
        CentralMEParameters.mu_length:    {'value': mu},
        SHO_Parameters.b_length: b_len,
    }
    CentralForce_JTScheme.turnDebugMode(False)
    CentralForce_JTScheme.setInteractionParameters(**kwargs)
    me = CentralForce_JTScheme(ket, bra)
    print(me.value)
    # me.saveXLog('FR_gauss_gR')
    
    