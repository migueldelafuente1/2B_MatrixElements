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
from helpers.Enums import PotentialForms, SHO_Parameters, BrinkBoekerParameters, \
    ForceEnum, CentralMEParameters, DensityDependentParameters
from matrix_elements.SpinOrbitForces import SpinOrbitForce, ShortRangeSpinOrbit_JTScheme,\
    SpinOrbitForce_JTScheme 
from helpers.Log import XLog
from matrix_elements.CentralForces import CoulombForce, KineticTwoBody_JTScheme,\
    DensityDependentForce_JTScheme, CentralForce_JTScheme
from helpers.TBME_SpeedRunner import TBME_SpeedRunner
from helpers.matrixElementHandlers import MatrixElementFilesComparator


if __name__ == "__main__":
    
    parsed_args = argv
    if len(parsed_args) > 1:
        # TODO: Define Parser and input process
        pass
    else:
        pass
        # TODO: Run the program from a file 'input.xml' next to the main
        
        #_runner = TBME_SpeedRunner(filename='input_B1.xml')
        # _runner = TBME_Runner(filename='input.xml')
        # _runner = TBME_Runner(filename='input.xml')
        # _runner = TBME_Runner(filename='input_D1S.xml')
        # _runner.run()
        
        _runner = TBME_SpeedRunner(filename='input_D1S.xml')
        # _runner = TBME_SpeedRunner(filename='input.xml')
        _runner.run()
        print(" The program has ended without incidences.")
        
        
    # kwargs = {
    #     SHO_Parameters.A_Mass       : 4,
    #     SHO_Parameters.Z            : 2,
    #     SHO_Parameters.b_length     : 1.5,
    #     SHO_Parameters.hbar_omega   : 18.4586,
    #
    #
    #     'part0':{
    #         'potential': 'gaussian', 'constant': '-1476.4145', 'mu_length': '0.0035'
    #     },'part1':{
    #         'potential': 'gaussian', 'constant': ' -13.6704', 'mu_length': '0.0080'
    #     },'part2':{
    #         'potential': 'gaussian', 'constant': '-141.0067', 'mu_length': '0.0181'
    #     },'part3':{
    #         'potential': 'gaussian', 'constant': ' -82.3763', 'mu_length': '0.0412'
    #     },'part4':{
    #         'potential': 'gaussian', 'constant': '  -4.3830', 'mu_length': '0.0936'
    #     },'part5':{
    #         'potential': 'gaussian', 'constant': ' -16.8006', 'mu_length': '0.2126'
    #     },'part6':{
    #         'potential': 'gaussian', 'constant': '  -3.7970', 'mu_length': '0.4828'
    #     },'part7':{
    #         'potential': 'gaussian', 'constant': '  -1.3423', 'mu_length': '1.0965'
    #     },'part8':{
    #         'potential': 'gaussian', 'constant': '  -1.1175', 'mu_length': '2.4906'
    #     },'part9':{
    #         'potential': 'gaussian', 'constant': '   0.0464', 'mu_length': '5.6569'}, 
    # }
    
    kwargs = {
        SHO_Parameters.A_Mass       : 16,
        SHO_Parameters.Z            : 8,
        SHO_Parameters.b_length     : 1.5, #1.4989,
        SHO_Parameters.hbar_omega   : 18.4586,
        
        #CentralMEParameters.potential : PotentialForms.Yukawa,
        #CentralMEParameters.mu_length : 2, 
        # CentralMEParameters.constant  : 1,
        # DensityDependentParameters.alpha : 1,
        # DensityDependentParameters.x0 : 1
        CentralMEParameters.potential : PotentialForms.Gaussian,
        CentralMEParameters.mu_length : 1.4,
        CentralMEParameters.constant  : -70.0
    }
    
    J = 1
    T = 0
    bra_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1), 
                                   QN_1body_jj(0,0,1), J, T)
    ket_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,1,1), 
                                   QN_1body_jj(0,1,3), J, T)
    
    CentralForce_JTScheme.turnDebugMode(True)
    CentralForce_JTScheme.setInteractionParameters(**kwargs)
    me = CentralForce_JTScheme(bra_, ket_)
    print("me: ", me.value)
    me.saveXLog('me_dens')
    # df = me.getDebuggingTable('me_{}_table.csv'.format('(2d2d_LS_2d2d)L2S1J2'))
    XLog.resetLog()
    
    # with open('results/BB_LS_SPSDPF_1.com', 'r') as f:
    #     data = f.readlines()[1:]
    #
    # k = 0
    # for i in range(len(data)):
    #     if k == 0:
    #         _, _, a, b, c, d, _, _ = data[i].split()
    #         Ls = []
    #         for x in (a, b, c, d):
    #             x = int(x)
    #             lx = (x - (x//10000))//100
    #             Ls.append(lx)
    #         if (Ls[0] + Ls[1]) % 2 != (Ls[2] + Ls[3]) % 2:
    #             print("Not good parity:", a, b, c, d) 
    #     elif k == 2:
    #         k = 0
    #         continue
    #     k += 1
    #
    # print('end')
    
    
    # kwargs[CentralMEParameters.n_power] = 0
    # ShortRangeSpinOrbit_JTScheme.turnDebugMode()
    # ShortRangeSpinOrbit_JTScheme.setInteractionParameters(**kwargs)
    # me = ShortRangeSpinOrbit_JTScheme(bra_, ket_)
    # result_ = me.value
    # print("me short: ", me.value)
    # me.saveXLog('me_shortLS')
    
    # kwargs = {
    #     SHO_Parameters.A_Mass       : 4,
    #     SHO_Parameters.b_length     : 1.4989,
    #     SHO_Parameters.hbar_omega   : 18.4586,
    #     BrinkBoekerParameters.mu_length : {'part_1': 0.7, 'part_2': 1.4},
    #     BrinkBoekerParameters.Wigner    : {'part_1': 0.0, 'part_2': -30.0},#{'part_1': 595.55, 'part_2': -72.21},
    #     BrinkBoekerParameters.Majorana  : {'part_1': 0.0, 'part_2': 0.0},#{'part_1': -206.05, 'part_2': -68.39},
    #     BrinkBoekerParameters.Bartlett  : {'part_1': 0.0, 'part_2': 0.0},
    #     BrinkBoekerParameters.Heisenberg: {'part_1': 0.0, 'part_2': 0.0}
    # }
    # BrinkBoeker.setInteractionParameters(**kwargs)
    #
    # BrinkBoeker.turnDebugMode(True)
    # me = BrinkBoeker(
    #     QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1), QN_1body_jj(0,0,1),1, 0),
    #     QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1), QN_1body_jj(0,0,1),1, 0),
    #     # QN_2body_jj_JT_Coupling(QN_1body_jj(0,1,1), QN_1body_jj(0,1,3),1, 0)
    # #     # QN_2body_jj_JT_Coupling(QN_1body_jj(2,0,1), QN_1body_jj(0,2,5), 3, 0),
    # #     # QN_2body_jj_JT_Coupling(QN_1body_jj(0,1,3), QN_1body_jj(1,3,5), 3, 0)
    # #     # QN_2body_jj_JT_Coupling(QN_1body_jj(0,2,3), QN_1body_jj(0,2,3), 2, 1), 
    # #     # QN_2body_jj_JT_Coupling(QN_1body_jj(0,2,3), QN_1body_jj(0,2,3), 2, 1)
    # #     # QN_2body_jj_JT_Coupling(QN_1body_jj(1,2,5), QN_1body_jj(1,3,5), 1, 0), 
    # #     # QN_2body_jj_JT_Coupling(QN_1body_jj(1,2,5), QN_1body_jj(1,3,5), 1, 0)
    #     )       
    # print("me: ", me.value)
    # me.saveXLog('me_test')
    #
    #
    # BrinkBoeker.turnDebugMode(False)
    # _runner = TBME_Runner(filename='input.xml')
    # _runner.run()
     
    
    #io.run_antoine_output('INPUT_P.txt')
