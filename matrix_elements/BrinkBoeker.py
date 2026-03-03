'''
Created on Feb 23, 2021

@author: Miguel
'''
import numpy as np

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    _TwoBodyMatrixElement_Antisym_JTCoupled
from matrix_elements.transformations import TalmiTransformation

from helpers.Enums import BrinkBoekerParameters as bb_p, CentralMEParameters,\
    PotentialForms, PotentialSeriesParameters
from helpers.Enums import AttributeArgs
from helpers.Enums import SHO_Parameters
from helpers.integrals import talmiIntegral

from helpers.Log import XLog
from helpers.Helpers import safe_wigner_6j, safe_clebsch_gordan, safe_wigner_9j,\
    safe_3j_symbols, fact, double_factorial, gamma_half_int

class BrinkBoeker_BrodyMoshinsky(_TwoBodyMatrixElement_JTCoupled, TalmiTransformation):
    
    """
    Implementation of the central force for two gaussians_, overriding of the
    BrodyMoschinsky transformation method to skip
    """
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ 
        Implement the parameters for the Brink-Boeker interaction calculation. 
        
        """
        # Refresh the Force parameters
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        _b = SHO_Parameters.b_length
        cls.PARAMS_SHO[_b] = float(kwargs.get(_b))
        
        part_1 = AttributeArgs.ForceArgs.Brink_Boeker.part_1
        part_2 = AttributeArgs.ForceArgs.Brink_Boeker.part_2
        
        cls.PARAMS_FORCE[0] = {}
        cls.PARAMS_FORCE[1] = {}
        
        for param in bb_p.members():
            cls.PARAMS_FORCE[0][param] = float(kwargs[param].get(part_1))
            cls.PARAMS_FORCE[1][param] = float(kwargs[param].get(part_2))
        
        cls.PARAMS_FORCE[CentralMEParameters.potential] = PotentialForms.Gaussian
        #cls.plotRadialPotential()
        
        cls.numberGaussians  = 2
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = ([], [])
    
    def _validKetTotalSpins(self):
        """ For Central Interaction, <S |Vc| S'> != 0 only if  S=S' """
        return (self.S_bra, )
    
    def _validKetTotalAngularMomentums(self):
        """ For Central Interaction, <L |Vc| L'> != 0 only if  L=L' """
        return (self.L_bra, )
    
    def _validKet_relativeAngularMomentums(self):
        """ Central interaction only allows l'==l"""
        return (self._l, )
    
    def _globalInteractionCoefficient(self):
        # no special interaction constant for the Central ME
        return 1
    
    def _interactionConstantsForCOM_Iteration(self):
        # no special internal c.o.m interaction constants for the Central ME
        return 1
    
    def deltaConditionsForGlobalQN(self):
        """ 
        Define if non null requirements on LS coupled J Matrix Element, 
        before doing the center of mass decomposition.
        
        NOTE: Redundant if run from JJ -> LS recoupling
        """
        if ((self.L_bra != self.L_ket)
            or (self.S_bra != self.S_ket)
            or (self._l != self._l_q)):
            
            if self.DEBUG_MODE:
                self.details = "deltaConditionsForGlobalQN = False central BB {}\n {}"\
                    .format(str(self.bra), str(self.ket))
            return False
        
        return True
    
    def _deltaConditionsForCOM_Iteration(self):
        """ condition for the antisymmetrization_  """
        if (((self.S_bra + self.T + self._l) % 2 == 1) and 
            ((self.S_ket + self.T + self._l_q) % 2 == 1)):
                return True
        return False
    
    @classmethod
    def _calculateIntegrals(cls, n_integrals=1):
        """
        >> Overwrite to have two parts
        """
        for p in range(cls._integrals_p_max + 1, 
                       cls._integrals_p_max + n_integrals +1):
            for part in range(cls.numberGaussians): 
                args = [
                    cls.PARAMS_FORCE.get(CentralMEParameters.potential),
                    cls.PARAMS_SHO.get(SHO_Parameters.b_length), # * np.sqrt(2), # 
                    cls.PARAMS_FORCE[part].get(CentralMEParameters.mu_length),
                    cls.PARAMS_FORCE[part].get(CentralMEParameters.n_power)
                ]
                
                cls._talmiIntegrals[part].append(talmiIntegral(p, *args))
            
            cls._integrals_p_max += 1
    
    def talmiIntegral(self):
        """ 
        >> Overwrite to have two parts
        Get or update Talmi integrals for the calculations
        """
        if self._p > self._integrals_p_max:
            self._calculateIntegrals(n_integrals = max(self.rho_bra, self.rho_ket, 1))
        return self._talmiIntegrals[self._part].__getitem__(self._p)
    
    def centerOfMassMatrixElementEvaluation(self):
        """
        Radial Brody-Moshinsky transformation, direct implementation for  
        central force.
        """
        return  self._BrodyMoshinskyTransformation()
    
    
    def _LScoupled_MatrixElement(self):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        aux_sum = 0.0
        # Sum of gaussians and projection operators
        for i in range(self.numberGaussians):
            self._part = i
            
            # Radial Part for Gaussian Integral
            radial_energy = self.centerOfMassMatrixElementEvaluation()
            
            # Exchange Part
            # W + P(S)* B - P(T)* H - P(T)*P(S)* M
            _S_aux = (-1)**(self.S_bra + 1)
            _T_aux = (-1)**(self.T)
            _L_aux = (-1)**(self.T + self.S_bra + 1)
            
            exchange_energy = (
                self.PARAMS_FORCE[i].get(bb_p.Wigner),
                self.PARAMS_FORCE[i].get(bb_p.Bartlett)   * _S_aux,
                self.PARAMS_FORCE[i].get(bb_p.Heisenberg) * _T_aux,
                self.PARAMS_FORCE[i].get(bb_p.Majorana)   * _L_aux
            )
            
            # Add up
            prod_part = radial_energy * sum(exchange_energy)
            aux_sum += prod_part
            
            if self.DEBUG_MODE:
                XLog.write('BB', radial=radial_energy, exch=exchange_energy, 
                           exch_sum=sum(exchange_energy), val=prod_part)
        
        return aux_sum 


class BrinkBoeker_Multipolar(_TwoBodyMatrixElement_Antisym_JTCoupled, 
                             BrinkBoeker_BrodyMoshinsky):
    
    
    K_max = 10
    
    def _LScoupled_MatrixElement(self):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        if (self.bra.shellStatesNotation == '0s1/2 0s1/2' and 
            self.ket.shellStatesNotation == '0p1/2 0p1/2'):
            _ = 0
        
        aux_sum = 0.0
        # Sum of gaussians and projection operators
        for i in range(self.numberGaussians):
            self._part = i
            
            # Radial Part for Gaussian Integral
            radial_energy = self.multipolarEvaluation()
            
            # Exchange Part
            # W + P(S)* B - P(T)* H - P(T)*P(S)* M
            _S_aux = (-1)**(self.S_bra + 1)
            _T_aux = (-1)**(self.T)
            _L_aux = (-1)**(self.T + self.S_bra + 1)
            
            exchange_energy = (
                self.PARAMS_FORCE[i].get(bb_p.Wigner),
                self.PARAMS_FORCE[i].get(bb_p.Bartlett)   * _S_aux,
                self.PARAMS_FORCE[i].get(bb_p.Heisenberg) * _T_aux,
                self.PARAMS_FORCE[i].get(bb_p.Majorana)   * _L_aux
            )
            
            # Add up
            prod_part = radial_energy * sum(exchange_energy)
            aux_sum += prod_part
            
            if self.DEBUG_MODE:
                XLog.write('BB', radial=radial_energy, exch=exchange_energy, 
                           exch_sum=sum(exchange_energy), val=prod_part)
        
        return aux_sum
    
    def _validKetTotalAngularMomentums(self):
        return (self.L_bra, )
    def _validKetTotalSpins(self):
        return (self.S_bra, )
    
    @classmethod
    def _setBiggerKMAX(cls, k_new):
        cls.K_max = max(k_new, cls.K_max)
    
    def multipolarEvaluation(self):
        """ 
        Evaluates  W_0(r1,r2) Y0*Y0 + W_1(r1,r2) Y1*Y1 + ...  matrix element
        """ 
        V_gauss = [0, ]*(self.K_max + 1)
        k_max = min(self.bra.l1 + self.ket.l1, self.bra.l2 + self.ket.l2)
        if k_max > self.K_max: self._setBiggerKMAX(k_max) 
        
        for k in range(self.K_max +1):            
            if ((self.bra.l1 + self.ket.l1 + k)%2 or (self.bra.l2 + self.ket.l2 + k)%2): 
                continue
            
            const_quad = 1 #0.6 if k == 2 else 1.0
                
            angular = self._angularCoefficient(k)
            if self.isNullValue(angular): continue
            
            radial  = self._radialIntegrals(k) #* self._radialIntegrals(k, 2) 
            
            V_gauss[k] = angular * radial * const_quad
        
        return sum(V_gauss)
    
    _angular_coeffs = {}
    @classmethod
    def _get_keyAngular(cls,  l1, l2, l1q, l2q, L, k):
        return "{}_{}_{}_{}_{}_{}".format(l1, l2, l1q, l2q, L, k)
    
    @classmethod
    def _save_angularCoeff(cls, l1, l2, l1q, l2q, L, k, value):
        key_ = cls._get_keyAngular(l1, l2, l1q, l2q, L, k)
        cls._angular_coeffs[key_] = value
    
    @classmethod
    def _get_angularCoeff(cls,  l1, l2, l1q, l2q, L, k):
        key_ = cls._get_keyAngular(l1, l2, l1q, l2q, L, k)
        if key_ in cls._angular_coeffs:
            return cls._angular_coeffs[key_]
        return None
    
    
    def _angularCoefficient(self, k):
        val_prev = self._get_angularCoeff(self.bra.l1, self.bra.l2, 
                                          self.ket.l1, self.ket.l2, self.L_bra, k)
        if val_prev: return val_prev
        
        factor = safe_wigner_6j(self.bra.l1, self.bra.l2, self.L_bra,
                                self.ket.l2, self.ket.l1, k)
        factor *= (-1)**(self.bra.l2 + self.ket.l1 + self.L_bra) #* (np.sqrt(2*self.L_bra + 1))
        
        aux = np.sqrt( (2*self.ket.l1 + 1)*(2*self.ket.l2 + 1)*
                      ((2*self.bra.l1 + 1)*(2*self.bra.l2 + 1))) * (2*k + 1) / (4*np.pi)
        aux *= safe_3j_symbols(self.bra.l1, k, self.ket.l1, 0, 0, 0)
        aux *= safe_3j_symbols(self.bra.l2, k, self.ket.l2, 0, 0, 0)
        aux *= (-1)**(self.bra.l1 + self.bra.l2)
        
        value = factor * aux 
        
        self._save_angularCoeff(self.bra.l1, self.bra.l2, 
                                self.ket.l1, self.ket.l2, self.L_bra, k, value)
        return value    
    
    
    S_RANGE_MAX = 30
    _radial_Coeffs = {}
    @classmethod
    def _radialAuxCoeff(cls, k, s, part):
        if not (part, k, s) in cls._radial_Coeffs:
            val = - gamma_half_int(2*(s + k) + 3) - fact(s)
            cls._radial_Coeffs[(part, k, s)] = np.exp(val) * np.sqrt(np.pi) / 2
        return cls._radial_Coeffs[(part, k, s)]
    
    def _radialIntegrals(self, k):
        sum_ = 0
        
        for s in range(self.S_RANGE_MAX +1):
            
            sum_s = [0, 0]
            for term in (1, 2):
                n ,l  = getattr(self.bra, f'n{term}'), getattr(self.bra, f'l{term}')
                nq,lq = getattr(self.ket, f'n{term}'), getattr(self.ket, f'l{term}')
                
                self._n   = n
                self._n_q = nq
                self._l   = l
                self._l_q = lq
                    
                sum_s[term - 1] = self._integrals1B(k,  s)
            
            coeff_s = self._radialAuxCoeff(k, s, self._part)
            
            sum_ += coeff_s * sum_s[0] * sum_s[1]
        
        return sum_ * (4*np.pi)
    
    def _normConstantSHO(self):
        factor = 2 / (self.PARAMS_SHO[SHO_Parameters.b_length]**3)
        aux = 0.5 * (fact( self._n) + fact( self._n_q)
                     - gamma_half_int(2*(self._n + self._l) + 3 )
                     - gamma_half_int(2*(self._n_q + self._l_q) + 3 ) 
        )
        return factor * np.exp(aux)  
    
    def _LaguerreCoefficient(self, m, n, l):
        
        aux = (
            gamma_half_int(2*(n + l) + 3 )
            - (gamma_half_int(2*(l + m) + 3 ) + fact(n - m) + fact(m))
        )
        return (-1)**m * np.exp(aux)
        
        
    def _integrals1B(self, k, s):
        
        b_len   = self.PARAMS_SHO.get(SHO_Parameters.b_length)
        mu_len  = self.PARAMS_FORCE[self._part].get(CentralMEParameters.mu_length)
        A       = (1/b_len**2) + (1/mu_len**2)
        
        KS = 2*s + k
        
        sum_ = 0
        for m1 in range(self._n +1):
            c1_ = self._LaguerreCoefficient(m1, self._n, self._l)
            for m2 in range(self._n_q +1):
                c2_ = self._LaguerreCoefficient(m2, self._n_q, self._l_q)
                
                K = 2*(m1 + m2) + (self._l + self._l_q) + KS + 2
                
                factor = (
                    gamma_half_int(K + 1)
                    + np.log(b_len / mu_len) * (KS)
                    - np.log(b_len) * (K - 2)
                    - np.log(A) * ((K + 1)/2)
                )
                
                sum_ += c1_ * c2_ * 0.5 * np.exp(factor)
        
        return sum_ * self._normConstantSHO()
    

class BrinkBoeker(
        BrinkBoeker_BrodyMoshinsky,
        # BrinkBoeker_Multipolar,
        ):
    pass


class PotentialSeries_JTScheme(BrinkBoeker):
    
    """
    The gaussian_ series are useful to mimic non analytically integrable
    potentials, this interaction is an extension of the Brink_Boeker for 
    indeterminate number of Wigner terms.
    """
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ 
        Implement the parameters for the Talmi Integrals. 
        """
        # Refresh the Force parameters
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        _b = SHO_Parameters.b_length
        cls.PARAMS_SHO[_b] = float(kwargs.get(_b))
        
        part    = PotentialSeriesParameters.part
        pot_key = CentralMEParameters.potential
        
        cls.numberGaussians = 0
        i=-1
        for param, values in kwargs.items():
            if not param.startswith(part):
                continue
            # i = int(param.split(part)[1])
            i += 1
            cls.PARAMS_FORCE[i] = {}
            potential = values.get(pot_key)
            cls.PARAMS_FORCE[i][pot_key] = potential
            for attr in (CentralMEParameters.mu_length, 
                         CentralMEParameters.constant,):
                ## set default geometric shape mu = Constant = 1 if not given.
                cls.PARAMS_FORCE[i][attr]  = float(values.get(attr, 1))
            for attr in (CentralMEParameters.opt_mu_2,
                         CentralMEParameters.opt_mu_3,
                         CentralMEParameters.opt_cutoff):
                if attr in values:
                    cls.PARAMS_FORCE[i][attr]  = float(values.get(attr))
            
            if potential in (PotentialForms.Power, 
                             PotentialForms.Gaussian_power,
                             PotentialForms.YukawaGauss_power):
                cls.PARAMS_FORCE[i][CentralMEParameters.n_power] = \
                    int(values.get(CentralMEParameters.n_power, 0))
            elif potential == PotentialForms.Wood_Saxon:
                A = float(kwargs.get(SHO_Parameters.A_Mass, 1))
                cls.PARAMS_FORCE[i][CentralMEParameters.opt_mu_2] *= (A**(1/3))
                
            cls.numberGaussians += 1
        
        #cls.plotRadialPotential()
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = tuple(([] for _ in range(cls.numberGaussians)))
        
    
    @classmethod
    def _calculateIntegrals(cls, n_integrals=1):
        """
        >> Overwrite to have N parts
        """
        for p in range(cls._integrals_p_max + 1, 
                       cls._integrals_p_max + n_integrals +1):
            
            for part in range(cls.numberGaussians):
                
                arg_keys = [
                    CentralMEParameters.potential, 
                    SHO_Parameters.b_length,
                    CentralMEParameters.mu_length,
                    CentralMEParameters.n_power
                ]
                args = [ 
                    cls.PARAMS_FORCE[part].get(arg_keys[0]), 
                    cls.PARAMS_SHO        .get(arg_keys[1]),# * np.sqrt(2), # 
                    cls.PARAMS_FORCE[part].get(arg_keys[2]), 
                    cls.PARAMS_FORCE[part].get(arg_keys[3]),
                ]
                kwargs = map(lambda x: (x, cls.PARAMS_FORCE[part].get(x, None)), 
                             CentralMEParameters.members(but=arg_keys))
                kwargs = dict(filter(lambda x: x[1] != None, kwargs))
                
                cls._talmiIntegrals[part].append(talmiIntegral(p, *args, **kwargs))
                
            cls._integrals_p_max += 1
    
    
    def _LScoupled_MatrixElement(self):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        aux_sum = 0.0
        # Sum of gaussians and projection operators
        for i in range(self.numberGaussians):
            self._part = i
            
            # Radial Part for Gaussian Integral
            radial_energy = self.centerOfMassMatrixElementEvaluation()
            
            if self.DEBUG_MODE:
                XLog.write('BB', mu=self.PARAMS_FORCE[i][CentralMEParameters.mu_length])
            
            prod_part = radial_energy * self.PARAMS_FORCE[i].get(CentralMEParameters.constant)
            aux_sum += prod_part
            
            if self.DEBUG_MODE:
                XLog.write('BB', radial=radial_energy, val=prod_part)
        
        return aux_sum 


class YukawiansM3Y_JTScheme(BrinkBoeker):
    
    """
    The gaussian_ series are useful to mimic non analytically integrable
    potentials, this interaction is an extension of the Brink_Boeker for 
    indeterminate number of Wigner terms.
    """
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ 
        Implement the parameters for the Talmi Integrals. 
        """
        # Refresh the Force parameters
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        _b = SHO_Parameters.b_length
        cls.PARAMS_SHO[_b] = float(kwargs.get(_b))
                
        cls.numberGaussians = 3
        for i in range(cls.numberGaussians):
            part_i = '{}_{}'.format(PotentialSeriesParameters.part, i+1)
            
            cls.PARAMS_FORCE[i] = {}
            
            for param in bb_p.members():
                cls.PARAMS_FORCE[i][param] = float(kwargs[param].get(part_i))
        
        cls.PARAMS_FORCE[CentralMEParameters.potential] = PotentialForms.Yukawa
                
        #cls.plotRadialPotential()
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = tuple(([] for _ in range(cls.numberGaussians)))
    

class YukawiansM3Y_tensor_JTScheme(YukawiansM3Y_JTScheme):
    """
    Introduced to speed up the tensor term.
    """
    _angular_com_ten = {}
    _angular_rel_ten = {}
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ 
        Implement the parameters for the Talmi Integrals. 
        """
        # Refresh the Force parameters
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        _b = SHO_Parameters.b_length
        cls.PARAMS_SHO[_b] = float(kwargs.get(_b))
                
        cls.numberGaussians = 2
        for i in range(cls.numberGaussians):
            part_i = '{}_{}'.format(PotentialSeriesParameters.part, i+1)
            
            cls.PARAMS_FORCE[i] = {}
            
            for param in bb_p.members():
                cls.PARAMS_FORCE[i][param] = float(kwargs[param].get(part_i))
        
        cls.PARAMS_FORCE[CentralMEParameters.potential] = PotentialForms.Yukawa
                
        #cls.plotRadialPotential()
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = tuple(([] for _ in range(cls.numberGaussians)))
    
    def deltaConditionsForGlobalQN(self):
        """ 
        Define if non null requirements on LS coupled J Matrix Element, 
        before doing the center of mass decomposition.
        
        NOTE: Redundant if run from JJ -> LS recoupling
        """
        if (abs(self.L_bra - self.L_ket) > 2):
            return False
        return True
    
    def _validKetTotalSpins(self):
        """ 
        Return ket states <tuple> of the total spin, for tensor force impose 
        S = S' = 1, return nothing to skip the bracket spin S=0
        """
        if self.S_bra == 0:
            return []
        return (1, )
    
    def _validKet_relativeAngularMomentums(self):
        """  Tensor interaction only allows l'=l and l +- 2 """
        if self._l > 1:
            return (self._l - 2, self._l, self._l + 2)
        else:
            return (self._l, self._l + 2)
    
    def _validKetTotalAngularMomentums(self):
        """ 
        Return ket states <tuple> of the total angular momentum, depending of 
        the Force.
        
        OJO: Moshinsky, lambda' = lambda, lambda +-1, lambda +-2!!! as condition
        in the C_Tensor, due rank 2 tensor coupling
        """
        _L_min = max(0, self.L_bra - 2)
        _L_max =        self.L_bra + 2
        gen_ = (l_q for l_q in range(_L_min, _L_max +1))
        return tuple(gen_)
    
    def _totalSpinTensorMatrixElement(self):
        """ <1/2 1/2 (S) | S^[1]| 1/2 1/2 (S)>, only non zero for S=S'=1 """
        if (self.S_bra == 0) or (self.S_ket == 0):
            return True, 0.0
        return False, 4.47213595499958 ## = np.sqrt(20)
    
    def centerOfMassMatrixElementEvaluation(self):
        """ 
        Radial Brody-Moshinsky transformation, implementation for a
        non central tensor force.
        """
        # the spin matrix element is 0 unless S=S'=1
        skip, spin_me = self._totalSpinTensorMatrixElement()
        if skip:
            return 0
        
        tpl_ = (self.L_bra, self.L_ket, self.J, self.T)
        if not tpl_ in self._angular_com_ten:
                        
            factor = safe_wigner_6j(self.L_bra, self.S_bra, self.J,
                                    self.S_ket, self.L_ket, 2)
            if not self.deltaConditionsForGlobalQN(): factor = 0
            
            phase   = (-1)**(self.S_bra + self.J + self.L_ket + self.L_bra)
            ## NOTE: the last L_bra should be from the ket since the W
            factor *= 3.8832518251113983 #* np.sqrt(2*self.J + 1) 
            ## 3.8832 = _sqrt(24*pi / 5)
            factor *= ((2*self.L_bra + 1)*(2*self.L_ket + 1))**0.5
    
            self._angular_com_ten[tpl_] = factor * spin_me * phase
        
        return self._angular_com_ten[tpl_] * self._BrodyMoshinskyTransformation()
    
    def _interactionConstantsForCOM_Iteration(self):
        
        tpl_ = (self.L_bra, self.L_ket, self._L, self._l, self._l_q)
        if not tpl_ in self._angular_rel_ten:
            factor = safe_wigner_6j(self._L, self._l,    self.L_bra,
                                    2,       self.L_ket, self._l_q)
            phase   = (-1)**(self._L + self._l)
            factor *= safe_clebsch_gordan(self._l, 2, self._l_q, 0, 0, 0)
            factor *= np.sqrt((2*self._l + 1)) * phase * 0.6307831305050401
            ## 0.6307 =  sqrt(5 / 4*pi)
            
            self._angular_rel_ten[tpl_] =  factor
        
        return self._angular_rel_ten[tpl_]

class YukawiansM3Y_SpinOrbit_JTScheme(YukawiansM3Y_tensor_JTScheme):
    """
    Introduced to speed up the Spin-Orbit term.
    """
    
    _angular_com_ls = {}
    _angular_rel_ls = {}
    
    def _validKet_relativeAngularMomentums(self):
        """  Spin orbit interaction only allows l'==l"""
        return (self._l, )
    
    def deltaConditionsForGlobalQN(self):
        """ 
        Define if non null requirements on LS coupled J Matrix Element, 
        before doing the center of mass decomposition.
        
        NOTE: Redundant if run from JJ -> LS recoupling
        """
        if (abs(self.L_bra - self.L_ket) > 1) :
            return False
        return True
    
    def _validKetTotalAngularMomentums(self):
        """ 
        Return ket states <tuple> of the total angular momentum, depending of 
        the Force.
        
        OJO: Moshinski, lambda' = lambda, lambda +- 1!!! as condition
        in the C_LS
        """
        _L_min = max(0, self.L_bra - 1)
        _L_max =        self.L_bra + 1
        gen_ = (l_q for l_q in range(_L_min, _L_max +1))
        return tuple(gen_)
    
    
    def _totalSpinTensorMatrixElement(self):
        """ <1/2 1/2 (S) | S^[1]| 1/2 1/2 (S)>, only non zero for S=S'=1 """
        if (self.S_bra != self.S_ket) or (self.S_bra == 0):
            return True, 0.0
        return False, 2.449489742783178 ## = np.sqrt(6)
    
    def centerOfMassMatrixElementEvaluation(self):
        """ 
        Radial Brody-Moshinsky transformation, implementation for a
        non central tensor force.
        """
        skip, spin_me = self._totalSpinTensorMatrixElement()
        if skip:
            return 0
        
        tpl_ = (self.L_bra, self.L_ket, self.J, self.T)
        if not tpl_ in self._angular_com_ls:
            factor = safe_wigner_6j(self.L_bra, self.S_bra, self.J,
                                    self.S_ket, self.L_ket,      1)
            if not self.deltaConditionsForGlobalQN(): factor = 0
            
            phase   = (-1)**(self.rho_bra + self.J)
            factor *= np.sqrt((2*self.L_bra + 1)*(2*self.L_ket + 1))
            
            self._angular_com_ten[tpl_] = factor * spin_me * phase
            
        return self._angular_com_ten[tpl_] * self._BrodyMoshinskyTransformation()
    
    def _interactionConstantsForCOM_Iteration(self):
        
        tpl_ = (self.L_bra, self.L_ket, self._L, self._l, self._l_q)
        if not tpl_ in self._angular_rel_ls:
            # no special internal c.o.m interaction constants for the Central ME
            factor = safe_wigner_6j(self._l,    self.L_bra, self._L, 
                                    self.L_ket, self._l_q,         1)
            
            factor *= np.sqrt(self._l * (self._l + 1) * (2*self._l + 1))
            self._angular_rel_ls[tpl_] = factor
        
        return self._angular_rel_ls[tpl_]
    