#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:57:34 2018

  * Generator of matrix elements in Antoine format

@author: miguel
"""

from sys import argv

from helpers.TBME_Runner import TBME_Runner
from helpers.WaveFunctions import QN_2body_L_Coupling, QN_1body_radial,\
    QN_2body_LS_Coupling, QN_1body_jj, QN_2body_jj_JT_Coupling,\
    QN_2body_jj_J_Coupling

from matrix_elements.BrinkBoeker import BrinkBoeker
from matrix_elements.TensorForces import TensorForce
from helpers.Enums import PotentialForms, SHO_Parameters, BrinkBoekerParameters
from matrix_elements.SpinOrbitForces import SpinOrbitForce, ShortRangeSpinOrbit_JTScheme
from helpers.Log import XLog
from matrix_elements.CentralForces import CoulombForce


if __name__ == "__main__":
    
    parsed_args = argv
    if len(parsed_args) > 1:
        # TODO: Define Parser and input process
        pass
    else:
        
#         sp_state_1 = QN_1body_jj(0, 0, 1)
#         sp_state_2 = QN_1body_jj(0, 0, 1)
#         sp_state_1_k = QN_1body_jj(0, 0, 1)
#         sp_state_2_k = QN_1body_jj(0, 0, 1)
#         
#         bra_ = QN_2body_jj_JT_Coupling(sp_state_1, sp_state_2, 0, 1)
#         ket_ = QN_2body_jj_JT_Coupling(sp_state_1_k, sp_state_2_k, 0, 1)
#         
#         bra_ = QN_2body_LS_Coupling(QN_1body_radial(1,4), 
#                                     QN_1body_radial(1,4), 2, 1)
#         ket_ = QN_2body_LS_Coupling(QN_1body_radial(1,4), 
#                                     QN_1body_radial(1,4), 2, 1)
#              
#         SpinOrbitForce.setInteractionParameters(
#             potential = PotentialForms.Power,
#             mu_length = 1,
#             n_power=0)
#         me = SpinOrbitForce(bra_, ket_, 3, run_it=True)
#         result_ = me.value
#         print("me: ", me.value)
#         df = me.getDebuggingTable('me_{}_table.csv'.format('(2d2d_Tensor_2d2d)L2S1J2'))
#         BrinkBoeker.setInteractionParameters()
        
        
        from helpers.Enums import ForceParameters, CentralMEParameters
        
        bra_ = QN_2body_jj_J_Coupling(QN_1body_jj(0,0,1, mt=1), 
                                      QN_1body_jj(0,0,1, mt=1), 0, 0)
        ket_ = QN_2body_jj_J_Coupling(QN_1body_jj(0,0,1, mt=1), 
                                      QN_1body_jj(0,0,1, mt=1), 0, 0)
        
        
        # CoulombForce.turnDebugMode()
        # CoulombForce.setInteractionParameters(
        #     **{SHO_Parameters.b_length : 1, SHO_Parameters.hbar_omega : 1})
        # me = CoulombForce(bra_, ket_)
        # print("me: ", me.value)
        # me.saveXLog('me_coul')
         
        kwargs = {
            SHO_Parameters.A_Mass       : 4,
            SHO_Parameters.b_length     : 1.4989,#1.2662,
            SHO_Parameters.hbar_omega   : 18.4586,
            BrinkBoekerParameters.mu_length : {'part_1': 0.7, 'part_2': 1.4},
            BrinkBoekerParameters.Wigner    : {'part_1': 0.0, 'part_2': -30.0},#{'part_1': 595.55, 'part_2': -72.21},
            BrinkBoekerParameters.Majorana  : {'part_1': 0.0, 'part_2': 0.0},#{'part_1': -206.05, 'part_2': -68.39},
            BrinkBoekerParameters.Bartlett  : {'part_1': 0.0, 'part_2': 0.0},
            BrinkBoekerParameters.Heisenberg: {'part_1': 0.0, 'part_2': 0.0}
        }
#         
#         ShortRangeSpinOrbit_JTScheme.setInteractionParameters(**kwargs)
#         
#         me = ShortRangeSpinOrbit_JTScheme(bra_, ket_, run_it=True)
#         result_ = me.value
#         print("me: ", me.value)

        BrinkBoeker.setInteractionParameters(**kwargs)
        
        BrinkBoeker.turnDebugMode(True)
        me = BrinkBoeker(
            QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1), QN_1body_jj(0,0,1),1, 0),
            QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1), QN_1body_jj(0,0,1),1, 0),
            # QN_2body_jj_JT_Coupling(QN_1body_jj(0,1,1), QN_1body_jj(0,1,3),1, 0)
        #     # QN_2body_jj_JT_Coupling(QN_1body_jj(2,0,1), QN_1body_jj(0,2,5), 3, 0),
        #     # QN_2body_jj_JT_Coupling(QN_1body_jj(0,1,3), QN_1body_jj(1,3,5), 3, 0)
        #     # QN_2body_jj_JT_Coupling(QN_1body_jj(0,2,3), QN_1body_jj(0,2,3), 2, 1), 
        #     # QN_2body_jj_JT_Coupling(QN_1body_jj(0,2,3), QN_1body_jj(0,2,3), 2, 1)
        #     # QN_2body_jj_JT_Coupling(QN_1body_jj(1,2,5), QN_1body_jj(1,3,5), 1, 0), 
        #     # QN_2body_jj_JT_Coupling(QN_1body_jj(1,2,5), QN_1body_jj(1,3,5), 1, 0)
            )       
        print("me: ", me.value)
        me.saveXLog('me_test')
        

        BrinkBoeker.turnDebugMode(False)
        _runner = TBME_Runner(filename='input.xml')
        _runner.run()
         
        
        #io.run_antoine_output('INPUT_P.txt')
    print(" The program has ended without incidences.")