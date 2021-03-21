#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:57:34 2018

  * Generator of matrix elements in Antoine format

@author: miguel
"""

from sys import argv

# import Input_Output as io
# import BM_brackets as bm
# import matrix_elements as me

from helpers.TBME_Runner import TBME_Runner
from helpers.WaveFunctions import QN_2body_L_Coupling, QN_1body_radial,\
    QN_2body_LS_Coupling, QN_1body_jj, QN_2body_jj_JT_Coupling

from matrix_elements.TensorForces import TensorForce
from helpers.Enums import PotentialForms
from matrix_elements.SpinOrbitForces import SpinOrbitForce


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
        
        from matrix_elements.BrinkBoeker import BrinkBoeker
        from helpers.Enums import SHO_Parameters
        
        ket_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,1,3), 
                                       QN_1body_jj(0,1,3), 1, 0)
        bra_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,1,1), 
                                       QN_1body_jj(0,1,1), 1, 0)
        
#         BrinkBoeker.setInteractionParameters(
#             SHO_Parameters.b_length = 1, SHO_Parameters.hbar_omega = 1)
        

        _runner = TBME_Runner(filename='input.xml')
        _runner.run()
         
        
        #io.run_antoine_output('INPUT_P.txt')
    print(" The program has ended without incidences.")