'''
Created on 7 nov 2023

@author: delafuente
'''
from helpers.WaveFunctions import QN_2body_L_Coupling, QN_1body_radial,\
    QN_1body_jj, QN_2body_jj_J_Coupling, QN_2body_jj_JT_Coupling

from helpers.Enums import PotentialForms, SHO_Parameters, BrinkBoekerParameters, \
    ForceEnum, CentralMEParameters, DensityDependentParameters,\
    SkyrmeBulkParameters
from helpers.Log import XLog
from matrix_elements.CentralForces import CoulombForce, KineticTwoBody_JTScheme,\
    CentralForce_JTScheme
from matrix_elements.DensityForces import DensityDependentForceFromFile_JScheme, \
    DensityDependentForce_JTScheme
from matrix_elements.SkyrmeForces import SkrymeBulk_JTScheme
from matrix_elements.SpinOrbitForces import SpinOrbitForce_JTScheme,\
    SpinOrbitFiniteRange_JTScheme

if __name__ == '__main__':
    
    option = 22
    
    if option == -1:
        ## Run a central interaction
        kwargs = {
            SHO_Parameters.A_Mass       : 4,
            SHO_Parameters.Z            : 2,
            SHO_Parameters.b_length     : 1.5,
            
            CentralMEParameters.constant  : 1.0,
            CentralMEParameters.potential : 'gaussian',
            CentralMEParameters.mu_length : 0.9,
            }
        
        J = 1
        T = 0
        bra_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,2,3), 
                                       QN_1body_jj(1,1,3), J, T)
        ket_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,2,5), 
                                       QN_1body_jj(1,1,3), J, T)
        
        CentralForce_JTScheme.turnDebugMode(True) # TensorForce_JTScheme
        CentralForce_JTScheme.setInteractionParameters(**kwargs)
        me_ = CentralForce_JTScheme(bra_, ket_)#, run_it=False)
        print("Gaussian ME:", me_.value)
        
    elif option == 0:
        kwargs = {
            SHO_Parameters.A_Mass       : 4,
            SHO_Parameters.Z            : 2,
            SHO_Parameters.b_length     : 1.5,
            SHO_Parameters.hbar_omega   : 18.4586,
        
        
            'part0':{
                'potential': 'gaussian', 'constant': '-1476.4145', 'mu_length': '0.0035'
            },'part1':{
                'potential': 'gaussian', 'constant': ' -13.6704', 'mu_length': '0.0080'
            },'part2':{
                'potential': 'gaussian', 'constant': '-141.0067', 'mu_length': '0.0181'
            },'part3':{
                'potential': 'gaussian', 'constant': ' -82.3763', 'mu_length': '0.0412'
            },'part4':{
                'potential': 'gaussian', 'constant': '  -4.3830', 'mu_length': '0.0936'
            },'part5':{
                'potential': 'gaussian', 'constant': ' -16.8006', 'mu_length': '0.2126'
            },'part6':{
                'potential': 'gaussian', 'constant': '  -3.7970', 'mu_length': '0.4828'
            },'part7':{
                'potential': 'gaussian', 'constant': '  -1.3423', 'mu_length': '1.0965'
            },'part8':{
                'potential': 'gaussian', 'constant': '  -1.1175', 'mu_length': '2.4906'
            },'part9':{
                'potential': 'gaussian', 'constant': '   0.0464', 'mu_length': '5.6569'}, 
        }
    elif option == 1:
    #--- TESTING INDIVIDUAL MATRIX ELEMENTS --------------------------------------- 
        kwargs = {
            SHO_Parameters.b_length  : 1.0, #1.4989,
            SkyrmeBulkParameters.t0  : 1.0, #
            SkyrmeBulkParameters.x0  : 0.0, #
            SkyrmeBulkParameters.t1  : 0.0, #
            SkyrmeBulkParameters.t2  : 0.0, #
        }
        
        J = 2
        T = 0
        bra_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,2,3), 
                                       QN_1body_jj(1,0,1), J, T)
        ket_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,2,5), 
                                       QN_1body_jj(1,0,1), J, T)
        
        SkrymeBulk_JTScheme.turnDebugMode(True)
        SkrymeBulk_JTScheme.setInteractionParameters(**kwargs)
        me = SkrymeBulk_JTScheme(bra_, ket_)
        print("me: ", me.value)
        


    
    elif option == 2:
        kwargs = {
            SHO_Parameters.b_length     : 1.5953,
            SHO_Parameters.A_Mass       : 16,
            SHO_Parameters.Z            : 8,
            # SHO_Parameters.hbar_omega   : 18.4586,
        
            DensityDependentParameters.alpha : 0.333333,
            DensityDependentParameters.x0    : 1.0,
            DensityDependentParameters.core  : {'core_b_len': 1.5953},
            DensityDependentParameters.constant :  1390.6, #1000.0, #
            DensityDependentParameters.file : 'final16OMz2_b15953_wf2.txt',
            DensityDependentParameters.integration : {'r_dim': 12, 'omega_ord': 14},
             #'final16OMz2_wf.txt', #'finalMz3_wf.txt', #
        
            # CentralMEParameters.potential : PotentialForms.Power,
            # CentralMEParameters.n_power   : 0,
            # CentralMEParameters.mu_length : 1.,
            # CentralMEParameters.constant  : 1.0
        }
        
        J = 1
        T = 0
        
        # bra_ = QN_2body_LS_Coupling(QN_1body_radial(2,0, mt= 1), 
        #                             QN_1body_radial(0,2, mt=-1), L, S)
        # ket_ = QN_2body_LS_Coupling(QN_1body_radial(0,1, mt= 1), 
        #                             QN_1body_radial(1,3, mt=-1), L, S)
        
        bra_ = QN_2body_jj_J_Coupling(QN_1body_jj(0,0,1, mt=-1), 
                                      QN_1body_jj(0,0,1, mt= 1), J, M=0)
        ket_ = QN_2body_jj_J_Coupling(QN_1body_jj(0,0,1, mt=-1), 
                                      QN_1body_jj(0,0,1, mt= 1), J, M=0)
        # bra_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1), 
        #                                QN_1body_jj(0,0,1), J, T)
        # ket_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1), 
        #                                QN_1body_jj(0,0,1), J, T)
        
        DensityDependentForceFromFile_JScheme.USING_LEBEDEV = True
        DensityDependentForceFromFile_JScheme.turnDebugMode(True)
        # OmegaOrd = 22
        # R_dim    = 30
        # DensityDependentForceFromFile_JScheme.setIntegrationGrid(R_dim, OmegaOrd)
        DensityDependentForceFromFile_JScheme.setInteractionParameters(**kwargs)
        me = DensityDependentForceFromFile_JScheme(bra_, ket_)
        
        print("me: ", me.value)
        me.saveXLog('me_DD_import')
        XLog.resetLog()
        
    elif option == 22:
        kwargs = {
            SHO_Parameters.b_length     : 1.5953,
            SHO_Parameters.A_Mass       : 16,
            SHO_Parameters.Z            : 8,
            # SHO_Parameters.hbar_omega   : 18.4586,
        
            DensityDependentParameters.alpha : 0.333333,
            DensityDependentParameters.x0    : 1.0,
            DensityDependentParameters.core  : {'core_b_len': 1.5953},
            DensityDependentParameters.constant :  1390.6, #1000.0, #
            # DensityDependentParameters.file : 'final16OMz2_b15953_wf2.txt',
            # DensityDependentParameters.integration : {'r_dim': 12, 'omega_ord': 14},
             #'final16OMz2_wf.txt', #'finalMz3_wf.txt', #
        
            # CentralMEParameters.potential : PotentialForms.Power,
            # CentralMEParameters.n_power   : 0,
            # CentralMEParameters.mu_length : 1.,
            # CentralMEParameters.constant  : 1.0
        }
        
        J = 1
        T = 0
        
        bra_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1), 
                                       QN_1body_jj(0,0,1), J, T)
        ket_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1), 
                                       QN_1body_jj(0,0,1), J, T)
        
        DensityDependentForce_JTScheme.turnDebugMode(True)
        # OmegaOrd = 22
        # R_dim    = 30
        DensityDependentForce_JTScheme.setInteractionParameters(**kwargs)
        me = DensityDependentForce_JTScheme(bra_, ket_)
        
        print("me: ", me.value)
        me.saveXLog('me_DD')
        XLog.resetLog()
        
    elif option == 3 : 
        
        ## TEST OF LS-MOSHINSKY AND MY LS TERMS
        
        kwargs_LS_mosh = {
            SHO_Parameters.b_length     : 1.8,
            SHO_Parameters.A_Mass       : 16,
            SHO_Parameters.Z            : 8,
        
            CentralMEParameters.potential : PotentialForms.Power,
            CentralMEParameters.n_power   : 0,
            CentralMEParameters.mu_length : 1.,
            CentralMEParameters.constant  : 1.0
        }
        
        kwargs_LS = {
            SHO_Parameters.b_length     : 1.8,
            SHO_Parameters.A_Mass       : 16,
            SHO_Parameters.Z            : 8,
        
            CentralMEParameters.potential : {'name': 'power'},
            CentralMEParameters.n_power   : {'value' : 0},
            CentralMEParameters.mu_length : {'value' : 1.0},
            BrinkBoekerParameters.Wigner  : {'value' : 1.0},
        }
        J = 2
        T = 1
        
        bra_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1, mt=-1), 
                                      QN_1body_jj(0,2,3, mt= 1), J, T)
        ket_ = QN_2body_jj_JT_Coupling(QN_1body_jj(0,1,1, mt=-1), 
                                      QN_1body_jj(0,1,3, mt= 1), J, T)
        
        SpinOrbitForce_JTScheme.turnDebugMode(True) # TensorForce_JTScheme
        SpinOrbitForce_JTScheme.setInteractionParameters(**kwargs_LS_mosh)
        me_1 = SpinOrbitForce_JTScheme(bra_, ket_)#, run_it=False)
        
        SpinOrbitFiniteRange_JTScheme.turnDebugMode(True) # TensorForce_JTScheme
        SpinOrbitFiniteRange_JTScheme.setInteractionParameters(**kwargs_LS)
        me_2 = SpinOrbitFiniteRange_JTScheme(bra_, ket_)#, run_it=False)
        
        
        
        # TensorS12_JTScheme.turnDebugMode(True) # TensorForce_JTScheme
        # TensorS12_JTScheme.setInteractionParameters(**kwargs)
        # me = TensorS12_JTScheme(bra_, ket_)#, run_it=False)
        # me.T = T
        print("me (bench): ", me_1.value)
        print("me (test): ",  me_2.value)
        # me.saveXLog('tens_S12')
        # me.saveXLog('me_ls1_import')
        # # df = me.getDebuggingTable('me_{}_table.csv'.format('(2d2d_LS_2d2d)L2S1J2'))
        # XLog.resetLog()
    
    
#------------------------------------------------------------------------------ 
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
     
