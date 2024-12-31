'''
Created on 3 dic 2024

@author: delafuente
'''
from helpers.Enums import CentralGeneralizedMEParameters, SHO_Parameters, BrinkBoekerParameters
from matrix_elements.CentralForces import CentralGeneralizedForce_JTScheme
from helpers.WaveFunctions import QN_2body_jj_JT_Coupling, QN_1body_jj

if __name__ == '__main__':
    
    " 0 5 001 001 101 101 0 1"
    J = 0
    T = 1
    bra = QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1, mt=-1),
                                  QN_1body_jj(0,1,1, mt=-1), J, T)
    ket = QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1, mt=-1),
                                  QN_1body_jj(0,1,1, mt=-1), J, T)
    # ket = QN_2body_jj_JT_Coupling(QN_1body_jj(0,1,1, mt=-1),
    #                               QN_1body_jj(0,0,1, mt=-1), J, T)
    
    kwargs = {
        CentralGeneralizedMEParameters.potential:   {'name': 'gaussian'},
        CentralGeneralizedMEParameters.constant :   {'value': 1,},
        CentralGeneralizedMEParameters.mu_length:   {'value': 10000,},
        CentralGeneralizedMEParameters.constant_R:  {'value': 1,},
        CentralGeneralizedMEParameters.mu_length_R: {'value': 1,},
        CentralGeneralizedMEParameters.potential_R: {'name': 'gaussian'},
        BrinkBoekerParameters.Wigner:     {'value': 1000,} ,
        BrinkBoekerParameters.Bartlett:   {'value': 0,},
        BrinkBoekerParameters.Heisenberg: {'value': 0,},
        BrinkBoekerParameters.Majorana:   {'value': 0,},
        SHO_Parameters.b_length: 1.8,
    }
    CentralGeneralizedForce_JTScheme.turnDebugMode(True)
    CentralGeneralizedForce_JTScheme.setInteractionParameters(**kwargs)
    me = CentralGeneralizedForce_JTScheme(ket, bra)
    print(me.value)
    me.saveXLog('FR_gauss_gR')