'''
Created on Oct 8, 2021

@author: Miguel
'''
from time import time
from copy import deepcopy
import numpy as np

from helpers.TBME_Runner import TBME_Runner, TBME_RunnerException
from helpers.Enums import InputParts as ip, ForceEnum, CouplingSchemeEnum
from helpers.Helpers import safe_wigner_9j, readAntoine
from matrix_elements import switchMatrixElementType
from helpers.WaveFunctions import QN_1body_jj, QN_2body_jj_J_Coupling,\
    QN_2body_jj_JT_Coupling
from itertools import combinations_with_replacement
from helpers.io_manager import valenceSpaceShellNames
from matrix_elements.MatrixElement import _TwoBodyMatrixElement

class TBME_SpdRunnerException(TBME_RunnerException):
    pass

class TBME_SpeedRunner(TBME_Runner):
    '''
    Efficient version of TBME_Runner, it can only use one time each force but 
    performs the valence space m.e. over a combination of LS matrix elements.
    
    This reduce the time spent in decoupling JT. Restricting the time only for 
    the two body LS m.e (with direct or implicit antisymetrization).
    '''

    isNullValue = _TwoBodyMatrixElement.isNullValue
    
    def __init__(self, filename='', verbose=True, manual_input={}):
        TBME_Runner.__init__(self, filename,  verbose, manual_input)
            
        # what's new??
        self.results_J  = {}
        self.results_JT = {}
        
        self.me_instances = []
        self.forces = []
        self.forcesDict = {}
        self.forcesIsAntisym = []
        self.forcesScheme    = []
        self.forcesNorms     = []
        self.valid_L_forKets = []
        self.valid_S_forKets = []
    
    def _checkHamilTypeAndForces(self):
        TBME_Runner._checkHamilTypeAndForces(self)
    
    def _setForces(self):
        
        _forcesAttr = ip.Force_Parameters
        
        sho_params = getattr(self.input_obj, ip.SHO_Parameters)
        
        for force, force_list in getattr(self.input_obj, _forcesAttr).items():
            if len(force_list) > 1:
                ## Verify forces do not repeat
                if force == ForceEnum.Force_From_File:
                    i = 0
                    for params in force_list:
                        force_str = force + str(i)
                        ## read from file case
                        self._readMatrixElementsFromFile(force_str, **params)
                else:
                    raise TBME_SpdRunnerException("Cannot compute two times "
                        "the same interaction: [{}] len={}".format(force, 
                                                                   len(force_list)))
            ## define interactions            
            self.forces.append(switchMatrixElementType(force))
            self.forcesDict[force] = len(self.forces) - 1
            ## unnecessary
            # self.tbme_class.resetInteractionParameters(also_SHO=True) 
            self.forces[-1].setInteractionParameters(**force_list[0], **sho_params)
            
            self.forcesIsAntisym.append(self.forces[-1].EXPLICIT_ANTISYMM)
            
            if self.forces[-1].COUPLING == CouplingSchemeEnum.JJ:
                self.forcesScheme.append(self._Scheme.J)
                dim_t = 6
            else:
                self.forcesScheme.append(self._Scheme.JT)
                dim_t = 2
            
            self.me_instances.append([None]*dim_t)
            self.valid_L_forKets.append([None]*dim_t)
            self.valid_S_forKets.append([None]*dim_t)
            self.forcesNorms.append([None]*dim_t)
            
            
        if self._com_correction:
            ## add JT kinetic term for calculation (the last)
            pass
        
    def run(self):
        ## 
        """
        Calculate all the matrix elements for all the interactions, and 
        print its combination in a file.
        """
        self._defineValenceSpaceEnergies()        
        self._checkHamilTypeAndForces()
        c_time = time()
        
        # get force from file if appears
        self._setForces()
        
        ## compute the valence space.   
            # the computation in J is common, separate between      
        self._computeForValenceSpaceJCoupled()
        print("Finished computation, Total time (s): [{}]".format(time() - c_time))
        
        self.resultsByInteraction['J_results']  = self.results_J 
        self.interactionSchemes['J_results']    = self._Scheme.J
        self.resultsByInteraction['JT_results'] = self.results_JT
        self.interactionSchemes['JT_results']   = self._Scheme.JT
        self.resultsByInteraction[ForceEnum.Kinetic_2Body] = self.com_2bme 
        self.interactionSchemes[ForceEnum.Kinetic_2Body]   = self._Scheme.JT
        
        self.combineAllResults()
        
        self.printMatrixElementsFile()
        self.printComMatrixElements()
    
    def _calculateCommonPhaseKet9j(self):
        """ Common phase for the direct antisymmetrized m.e. """
        pwr = 2*(self.ket_1.j + self.ket_2.j) + self.ket_1.l + self.ket_2.l
        self._common_phs_9j = (-1)**(pwr + self.J + 1)
    
    def _computeForValenceSpaceJCoupled(self):
        """ 
        method to run the whole valence space m.e. in the J scheme
        Indexing for J row:
            {0: pppp, 1:pnpn, 2:pnnp, 3:nppn, 4:npnp, 5:nnnn}
        """
        q_numbs = getattr(self.input_obj, ip.Valence_Space)
        self.valence_space = valenceSpaceShellNames(q_numbs)
        
        q_numbs = map(lambda qn: int(qn), q_numbs)
        q_numbs = sorted(q_numbs)#, reverse=True)
        q_numbs = list(combinations_with_replacement(q_numbs, 2))
        
        self._count = 0
        self._total_me = len(q_numbs)*(len(q_numbs)+1)//2
        for i in range(len(q_numbs)):
            self.bra_1 = QN_1body_jj(*readAntoine(q_numbs[i][0], l_ge_10=True))
            self.bra_2 = QN_1body_jj(*readAntoine(q_numbs[i][1], True))
            
            self.results_J [q_numbs[i]] = {}
            self.results_JT[q_numbs[i]] = {}
            self.com_2bme[q_numbs[i]]   = {}
            
            J_bra_min = abs(self.bra_1.j - self.bra_2.j) // 2
            J_bra_max = (self.bra_1.j + self.bra_2.j) // 2
            
            for j in range(i, len(q_numbs)):
                self._qqnn_curr = q_numbs[i], q_numbs[j]
                self.ket_1 = QN_1body_jj(*readAntoine(q_numbs[j][0], True))
                self.ket_2 = QN_1body_jj(*readAntoine(q_numbs[j][1], True))
                
                J_min = max(abs(self.ket_1.j - self.ket_2.j) // 2, J_bra_min)
                J_max = min((self.ket_1.j + self.ket_2.j) // 2,    J_bra_max)
                
                # define all elements to zero (LS results sum to them)
                mt_j_aux = dict([(mt, 0) for mt in self._JSchemeIndexing.keys()])
                aux_j = [(J, deepcopy(mt_j_aux)) for J in range(J_min, J_max +1)]
                aux_j    = dict(aux_j)
                
                aux_jt   = dict([(J, 0) for J in range(J_min, J_max +1)])
                self.results_J [q_numbs[i]][q_numbs[j]] = deepcopy(aux_j)
                self.results_JT[q_numbs[i]][q_numbs[j]] = {0: deepcopy(aux_jt), 
                                                           1: deepcopy(aux_jt)}
                self.com_2bme[q_numbs[i]][q_numbs[j]]   = {0: deepcopy(aux_jt), 
                                                           1: deepcopy(aux_jt)}
                self._count += 1
                self._tic = time()
                
                for J in range(J_min, J_max +1):
                    self.J = J
                    self._calculateCommonPhaseKet9j()
                    
                    self._evaluateMatrixElementValues_recoplingJtoLS()
    
    def _evaluateMatrixElementValues_recoplingJtoLS(self):
        """ 
        Method recopules the matrix element to access the _LS_coupled matrix
        element. Both for J scheme and JT scheme matrix elements.
        """
        self._phs_exch_J = 1 # TODO: Is the same for all mt states ??? j1+j2+J
        self._phs_exch_T = [1, 1]
        self._phase_9j   = 1
        
        bra, ket = self._qqnn_curr
        if bra == (1, 1) and ket == (101, 101):
            _ = 0
        for force, f in self.forcesDict.items():
            
            if self.forcesScheme[f] == self._Scheme.J:
                self._instance_J_SchemeWF(f)
            else:
                self._instance_JT_SchemeWF(f)
        
        self._LS_recoupling_ME()
        ## TODO: multiply here the normalization
    
    def _instance_J_SchemeWF(self, force_index):
        f = force_index
        ## TODO: remove self.bra and ket if is not necessary out of this method
        for m, mts in self._JSchemeIndexing.items():
            #tic = time()
            
            self.bra_1.m_t = mts[0]
            self.bra_2.m_t = mts[1]
            self.ket_1.m_t = mts[2]
            self.ket_2.m_t = mts[3]
            
            braJ = QN_2body_jj_J_Coupling(self.bra_1, self.bra_2, self.J)
            ketJ = QN_2body_jj_J_Coupling(self.ket_1, self.ket_2, self.J)
            
            self.me_instances[f][m] = self.forces[f](braJ, ketJ, run_it=False)
            self.forcesNorms[f][m]  = braJ.norm() * ketJ.norm() 
        
        phs, _ = ketJ.exchange()
        self._phs_exch_J = phs
    
    def _instance_JT_SchemeWF(self, force_index):
        f = force_index
        ## TODO: remove self.bra and ket if is not necessary out of this method
        for T in (0, 1):
            # assume M.E. cannot couple <(JT)|V|J'T'> if J', T' != J, T
            bra = QN_2body_jj_JT_Coupling(self.bra_1, self.bra_2, self.J, T)
            ket = QN_2body_jj_JT_Coupling(self.ket_1, self.ket_2, self.J, T)
            
            phs, _ = ket.exchange()
            self._phs_exch_T[T] = phs
            self.forcesNorms[f][T]  = bra.norm() * ket.norm()
            
            self.me_instances[f][T] = self.forces[f](bra, ket, run_it=False)
    
    def _angularRecouplingCoefficiens(self):
        """ 
        recycled from Jcoupled Matrix elements   
            Return
        :null_values          <bool>  To skip Moshinky transformation
        :recoupling_coeff     <float> 
        """
        # j attribute is defined as 2*j
        self.recoupling = 0.0
        
        w9j_bra = safe_wigner_9j(
            self.bra_1.l, 0.5, self.bra_1.j / 2,
            self.bra_2.l, 0.5, self.bra_2.j / 2,
            self.L_bra, self.S_bra, self.J)        

        if not self.isNullValue(w9j_bra):
            recoupling = ((self.bra_1.j + 1)*(self.bra_2.j + 1))**0.5 * w9j_bra
            
            w9j_ket = safe_wigner_9j(
                self.ket_1.l, 0.5, self.ket_1.j / 2,
                self.ket_2.l, 0.5, self.ket_2.j / 2,
                self.L_ket, self.S_ket, self.J)
            
            if not self.isNullValue(w9j_ket):
                recoupling *= w9j_ket
                recoupling *= ((self.ket_1.j + 1)*(self.ket_2.j + 1))**0.5
                recoupling *= ((2*self.S_bra + 1)*(2*self.L_bra + 1)
                                *(2*self.S_ket + 1)*(2*self.L_ket + 1))**0.5
                
                return (False, recoupling)
        return (True, 0.0)
    
    def _setTotalQQNNinME(self, qn=None):
        """ Auxiliary function to set the S and L momentum in the mmee_ 
        :qn (quantum number): 'S    or 'L'  
        """
        # TODO: remove assertion after Testing
        assert qn in ('S', 'L'), "invalid argument: S or L"
        valids = []
        for f in range(len(self.me_instances)):
            for t in range(len(self.me_instances[f])):
                if   qn == 'S':
                    self.me_instances[f][t].S_bra = self.S_bra
                    
                    valid = self.me_instances[f][t]._validKetTotalSpins()
                    self.valid_S_forKets[f][t] = valid
                    valids += valid
                    
                elif qn == 'L':
                    self.me_instances[f][t].L_bra = self.L_bra
                    
                    valid = self.me_instances[f][t]._validKetTotalAngularMomentums()
                    self.valid_L_forKets[f][t] = valid
                    valids += valid
                else:
                    raise TBME_SpdRunnerException("Invalid values for qn")

        if qn == 'S':
            self._validKetTotalSpins = set(valids)
        elif qn == 'L':
            self._validKetTotalAngularMomentums = set(valids)
        
    def _LS_recoupling_ME(self):
        """ 
        Obtains the non antisymmetrized matrix elements by recoupling to total
        L and S and call the Inner Interaction recoupled to LS - T scheme.
        """
        #sum_ = 0.
        
        L_max = self.bra_1.l + self.bra_2.l
        L_min = abs(self.bra_1.l - self.bra_2.l)
        
        for S_bra in (0, 1):
            self.S_bra = S_bra
            self._setTotalQQNNinME(qn='S')
            
            for S in self._validKetTotalSpins:
                self.S_ket = S
                
                for L_bra in range(L_min, L_max +1):
                    self.L_bra = L_bra
                    self._setTotalQQNNinME(qn='L')
                    
                    for L in self._validKetTotalAngularMomentums:
                        self.L_ket = L
                        
                        null, coupling = self._angularRecouplingCoefficiens()
                        if null:
                            continue
                        self.recoupling = coupling
                        self._phase_9j = self._common_phs_9j * ((-1)**(S + L))
                        
                        self._antisymmetrized_LS_element()
        
    
    def __auxSkipCurrentME(self, f, t):
        """ auxiliary function to skip invalid matrix elements """
        if self.me_instances[f][t].isNullMatrixElement:
            return True
        # if self.isNullValue(self.forcesNorms[f][t]):
        #     return True  
        if self.S_ket not in self.valid_S_forKets[f][t]:
            return True
        if self.L_ket not in self.valid_L_forKets[f][t]:
            return True
        
        return False
        
    def _antisymmetrized_LS_element(self):
        """
        Mediator function to evaluate the direct and exchange LS m.e, 
        This is an explicitly antysimetrized_ m.e, it evaluates the both direct
        and exchange matrix elements in the LS scheme.
        """
        bra, ket = self._qqnn_curr
        
        all_null = True
        _last = len(self.me_instances) - 1
        for f in range(len(self.me_instances)):
            if f == _last:
                _ = 0
            for t in range(len(self.me_instances[f])):
                
                if self.__auxSkipCurrentME(f, t):
                    continue
                
                # fixes the S / L for the ket
                self.me_instances[f][t].S_ket = self.S_ket
                self.me_instances[f][t].L_ket = self.L_ket
                
                direct = self.me_instances[f][t]._LScoupled_MatrixElement()                               
                exch_val = 0
                # explicit antysimmetric_ m.e. has exch_2bme set by constructor
                if self.forcesIsAntisym[f]:
                    ## perform exchange of the qqnn 
                    exch_2bme = self.me_instances[f][t].getExchangedME()
                        
                    exch_2bme.S_bra = self.S_bra
                    exch_2bme.S_ket = self.S_ket
                    exch_2bme.L_bra = self.L_bra
                    exch_2bme.L_ket = self.L_ket
                
                    exch_val = exch_2bme._LScoupled_MatrixElement()
                
                if self.forcesScheme[f] == self._Scheme.J:
                    # will exchange only if run trough 
                    exch_val *= self._phase_9j * self._phs_exch_J
                    value = self.forcesNorms[f][t] * (direct - exch_val) * self.recoupling
                    self.results_J [bra][ket][self.J][t] += value
                else:
                    exch_val *= self._phase_9j * self._phs_exch_T[t]
                    value = self.forcesNorms[f][t] * (direct - exch_val) * self.recoupling
                    
                    if f == _last and self._com_correction:
                        # if com_correction request, last m.e takes it
                        self.com_2bme[bra][ket][t][self.J]   += value
                    else:
                        self.results_JT[bra][ket][t][self.J] += value
                    
                all_null = all_null and self.isNullValue(value)
                    
                ## TODO: print times/ count progress
        if not all_null and self.PRINT_LOG:
            print(' * me[{}/{}]_({:.4}s): <{}|V|{} (J:{})>'
                  .format(self._count, self._total_me, time() - self._tic, 
                          bra, ket, self.J))
            # print('\t= {:.8} '.format(me.value))
    