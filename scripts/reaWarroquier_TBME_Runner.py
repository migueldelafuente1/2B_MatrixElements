'''
Created on 18 jun 2025

@author: delafuente

This module computes efficiently the matrix elements for the rearrangement following
the equiations on 
"
Warokquier, et al. Rearrangement Effects in Shell-Model Calculations Using 
                    Density-Depedent Interactions (1987)
Phys Reports 148, No 5 (1987) 249-306. North-Holand Amsterdam
"

The matrix elements in this work are in particle-hole formalism and require the
pre-calculation of three sub matrix elements related to the density and its 
derivatives.

The suite also requires the application over a spherical wave-function, computing
different functions of U,Vs with the matrix elements:
    Q(abcd; J) =  
        A1(U,V;abcd) * <ab|vDD|cd>_J
        + A21 * F(abcd; J) + A22 * F(abcd; J)
        + A3  * B(abcd;J) * CHI(U,V, <ab|d vDD phi(cd)|ee>_J=0>)
        + <if J=0>
            + A4  * B(abcd) * PI(U,V, <ee|dd vDD phi(ab)phi(cd)|ff>_J=0>) + perm.
    
    With F(abcd; J) being a function of the particle-hole <ab|vDD|cd>_J mm.ee.

For all of this, It makes no sense to implement this matrix element as in the 
other interactions.

Requirements:
DONE: Importing of a WF in U,V, verification of spherical conditions for the script.
DONE: Implememt a PANDYA transformation for particle-hole matrix elements
DONE: Implement the different DD matrix elements, for spherical symmetry but 
    with the importing of the WF from TAURUS txt file. Also generalized to x0H x0M

    UPDATE 1/8/25: CLOSED - DEPRECATED -------------------------------------

        There is no way to complement the rearrangement effects with the method
    in the article. It could be that the method is only applicable to RPA, it 
    could be that it requires the effective-hamiltonian renormalization interaction
    shit on the second part, or who knows...
        Anyway, with our method could not reproduce the rearrangement contributions
    using its method.
       ---------------------------------------------------------------------

'''
import numpy as np
import os
import shutil

from helpers.Enums import BrinkBoekerParameters, PotentialForms, SHO_Parameters,\
    DensityAndExchangeParameters
from helpers.Helpers import Constants, almostEqual, readAntoine,\
    getJrangeFor2ShellStates, safe_wigner_9j, safe_clebsch_gordan,\
    safe_wigner_6j
from helpers.WaveFunctions import QN_1body_jj, QN_2body_jj_JT_Coupling,\
    QN_2body_jj_J_Coupling
import time
from helpers.io_manager import TBME_Reader

_DD_pe = DensityAndExchangeParameters

from helpers import SCIPY_INSTALLED, MATPLOTLIB_INSTALLED
from matrix_elements.MatrixElement import MatrixElementException
from copy import copy, deepcopy
from helpers.mathFunctionsHelper import _buildRadialFunctionsArray,\
    _buildAngularYCoeffsArray, angular_Y_KM_index, sphericalHarmonic,\
    _angular_Y_KM_memo_accessor, _angular_Y_KM_me_memo
if MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt

class RearrangementTBMESpherical_Runner():
    
    _R_DIM = 15
    _A_DIM =  0
    _OMEGA = 16
    USING_LEBEDEV = True
    NULL_TOLERANCE = 1.0e-12
    OCCUPIED_TOL = 0.90
    APPLY_PH_OCCUPATION_DD_2BME = False
    
    _TESTING_CONST = 1 # 1e+18
    _F_REA_2EVAL   = [1, 1, 1]
    @classmethod
    def __new__(cls, *args, **kwargs):
        """
        Defining the class method attributes
        """
        cls.B_LEN    = 1
        cls.x0       = 1
        cls.x0H      = 0
        cls.x0M      = 0
        cls.ALPHA    = 1
        cls.CONST_t0 = 1
        cls.filename = None
        
        cls.A = 0
        cls.Z = 0
        cls.N = 0
        
        cls.sh_dim = 0
        cls.sh_states = []
        cls.sh_states_obj = {}
        cls.sh_states_sorted = [] ## |ab> 11, 12, 13, ..., 22, 23, ...
        cls.occupied_states  = {1:{}, -1:{}} ## 1=p,-1=n  1 = True... 4 = False if |V|< TOL
        
        cls.j_max = 0
        cls.n_max = 0
        cls.l_max = 0
        
        cls.U_sh = {1: {}, -1:{}} ## [p=1 {sh:}, n=-1 {sh:}]
        cls.V_sh = {1: {}, -1:{}} ## [p=1 {sh:}, n=-1 {sh:}]
        
        cls.sp_dim = 0
        cls.sp_states = []
        cls.U_sp = None
        cls.V_sp = None
        cls._density_matrix = None 
        cls._kappa_matrix   = None
        
        cls._radial_2b_wf_memo  = {}
        cls._sph_harmonic_memo  = {}
        
        cls._spatial_dens = None
        cls._spatial_dens_pn   = [None, None]
        cls._spatial_dens_imag = None ## test
        cls._spatial_dens_alp  = None
        cls._spatial_dens_alpM1  = None
        cls._spatial_dens_alpM2  = None
        
        instance = super().__new__(cls)
        return instance 
    
    def __init__(self):
        
        self._Q_ph_hamil = {}
        self._Q_final_hamil = [{}, {}]  ## ph, final
        
        self._PiZeta_hamil  = [{}, {}]  ## Pi(abcd, J=0) m.e. (1st derivative), saved with exchange me.
                                        ## Zeta(abcd, J=0) m.e. (2st derivative)
        self._P_hamil = {}
        self._Z_hamil = {}
        
        self._dd_hamil = [
            {},       # <ab|vDD|cd>_J
            {},       # <ab|d vDD phi(cd)|ee>_J=0>
            {},       # <ee|dd vDD phi(ab)phi(cd)|ff>_J=0>
            {},       # F matrix elements, saved with exchanged me.  <ab|vDD|cd>_J (ph)
            #{},       # F Rearr. matrix elements. 
        ]
        
        self.bra : QN_2body_jj_JT_Coupling = None
        self.ket : QN_2body_jj_JT_Coupling = None
        self.fix_1 = None # (qn Antoine, qn Antoine)
        self.fix_2 = None # (qn Antoine, qn Antoine)
        
        self._2b_sort_states    = []
        self._ph_sort_states    = []
        self._ph_sort_exch      = [] # Being exchanged, require to add permutations
        self._ph_sort_exch_P    = [] # Being exchanged, require to add permutations
        self._ph_sort_pn_dir    = []
        self._ph_sort_pn_exch_F = []
        self._ph_sort_pn_exch_P = []
        
        self.__radial_integrals_abcd = dict()
        self.__radial_int_am1PN_abcd = [dict(), dict()]
        self.__radial_int_am2PN_abcd = [dict(), dict()]
        self.__radial_int_am1PN_abcdef = dict()
        self.__radial_int_am2PN_abcdef = dict()
        
        self.__CGC_jab_byJ = {} ## CGC [J][M=-1,0,1][(ja, jb)]
        
        self._ME_TYPE = 0 # selector for ME evaluation - storage (0 dd, 1 Der1-dd, 2Der2-dd)
    
    #===========================================================================
    # # MAIN SCRIPT 
    #===========================================================================
    
    def run(self):
        ## initialization
        self._readWaveFunctionFromFile()
        self._setUpSHOBasisRadialAngular()
        self._setUpDensities()
        self._sortHamiltonianStatesFor2BandPH()
        self._setUpHamiltoniannStatesAndAngularFunctions()
        
        self.printUVs()
        
        print("[    ] Computation of JT DD matrix elements.")
        ## **
        if self.APPLY_PH_OCCUPATION_DD_2BME:
            self._calculateDDMatrixElements()
        
        if not almostEqual(self.x0, 1):
            self._calculateDDMatrixElements_2ndpart()
            self._calculateDDMatrixElements_3rdpart()
        self._calculateDDhamilForFparticleHole()
        self._calculateDDhamilForFparticleHole_PNPN()
        
        self._convertThePNPN_phMatrixElements()
        for II in range(4): 
            self._ME_TYPE = II
            self._exportHamilsDDTesting(tail='_ph')
        print("[DONE] Computation of JT DD matrix elements.")
        
        ## PH-matrix element computing
        ##     v----  This method (and subroutines) is obsolete, job done in **
        #self._calculateBulkMatrixElements() 
        self._computeRearrangementMatrixElements()
        self._Q_final_hamil[0] = deepcopy(self._Q_ph_hamil)
        
        self._convertFinalPHMatrixElements()
        
        ## Append 2b-DD matrix elements
        self._resetInternalHamiltonians(_dd_hamil=True)
        self._calculateDDMatrixElements_2bme()
        self._ME_TYPE = 0
        self._exportHamilsDDTesting(tail='_2b')
        self._appendDD2bMEToQMatrixElements()
        
        self._exportQHamiltonian()
    
    #===========================================================================
    
    def getMatrixElements(self, particle_hole_format=False):
        """
        Return the particle-particle interacting m.e. applying PANDYA from the
        pre-calculated particle-hole m.e.
        """
        if particle_hole_format:
            return self.self._Q_final_hamil[0]
        return self._Q_final_hamil[1]       
        
    def printUVs(self):
        """  Indicate the results for the spherical U-V matrices """
        print()
        print("   i   indx_sh  =      Up             Un             Vp          Vn  ")
        print("-"*70)
        for i, a in enumerate(self.sh_states):
            arg = [
                f"{x: >10.3e}" for x in 
                (self.U_sh[1][a], self.U_sh[-1][a], self.V_sh[1][a], self.V_sh[-1][a])
            ]
            arg[0] += "({:})".format(int(self.occupied_states[ 1][a]))
            arg[1] += "({:})".format(int(self.occupied_states[-1][a]))
            print(f" ({i: >2})  [{a: >6}] =", '  '.join(arg))
        print()
        
    def isNullValue(self, value):
        """ Method to fix cero values and stop calculations."""
        return abs(value) < self.NULL_TOLERANCE
    
    @staticmethod
    def _getJRangeForBraKet(bra, ket):
        """ Get the valid J range for two states, verifying also the parity"""
        
        Jmin0, Jmax0 = getJrangeFor2ShellStates(*bra)
        ## J range check
        Jcd_range = getJrangeFor2ShellStates(*ket)
        if (Jcd_range[0] > Jmax0) or  (Jcd_range[1] < Jmin0): 
            return 0, -1, False
        Jmin, Jmax = max(Jmin0, Jcd_range[0]), min(Jmax0, Jcd_range[1])
        
        ll_ = [readAntoine(x, True)[1] for x in (*bra, *ket)]
        parity_ok = sum(ll_) % 2 == 0
        
        return Jmin, Jmax, parity_ok
    
    def _sortHamiltonianStatesFor2BandPH(self):
        """
        To have a complete conversion of the 2.body hamiltonian 
        (for all the combinations of states for a SM calculation)
            
            Q(2b)_abcd_J = Pandya()_J' Q(ph)_adcb_J'
            
        The PH elements requires additionally the states for the pn-pn
        from the pp-nn ones, which get cumbersome for the exchanged options.
            
                    pn-pn PH   from pp-nn        
            DIR:     <ad, cb>  <-  <ac, db>
            EXCH F:  <ad, bc>  <-  <ab, dc>
            EXCH P:  <cb, ad>  <-  <ca, bd>
        """
        for ia, a in enumerate(self.sh_states):
            for b in self.sh_states[ia:]:
                bra = (a, b)
                for ic, c in enumerate(self.sh_states):
                    for d in self.sh_states[ic:]:
                        ket = (c, d)
                        bra_ph,    ket_ph    = (a, d), (c, b)
                        bra_pn_d,  ket_pn_d  = (a, c), (d, b)
                        bra_pn_eF, ket_pn_eF = (a, b), (d, c)
                        bra_pn_eP, ket_pn_eP = (c, a), (b, d)
                        bra_ph_e,  ket_ph_e  = (a, d), (b, c)
                        bra_ph_eP, ket_ph_eP = (c, b), (a, d)
                                              
                        self._2b_sort_states.append( (bra, ket) )
                        self._ph_sort_states.append( (bra_ph, ket_ph) )
                        self._ph_sort_pn_dir.append( (bra_pn_d,  ket_pn_d) )
                        self._ph_sort_pn_exch_F.append( (bra_pn_eF, ket_pn_eF) )
                        self._ph_sort_pn_exch_P.append( (bra_pn_eP, ket_pn_eP) )
                        
                        self._ph_sort_exch  .append( (bra_ph_e,  ket_ph_e) ) 
                        self._ph_sort_exch_P.append( (bra_ph_eP, ket_ph_eP) )
    
    def _resetInternalHamiltonians(self, 
                                   _dd_hamil=False, _P_hamil=False, _Z_hamil=False):
        """
        Reset the different internal hamiltonians
        """
        
        zeros_ = [0.0 for t in range(6)]
        
        _SET_qqnn = set(self._ph_sort_states + self._ph_sort_pn_dir)
        for bra, ket in _SET_qqnn:
            Jmin,  Jmax,  parOK  = self._getJRangeForBraKet(bra, ket)
            if not parOK: continue
            
            hamil_sect = dict([(J, deepcopy(zeros_))for J in range(Jmin, Jmax+1)])
            if _dd_hamil:
                if not bra in self._dd_hamil[0]:
                    self._dd_hamil[0][bra] = {}
                    self._dd_hamil[3][bra] = {}
                
                self._dd_hamil[0][bra][ket] = deepcopy(hamil_sect)
                self._dd_hamil[3][bra][ket] = deepcopy(hamil_sect)
            if _P_hamil:
                if not bra in self._P_hamil:
                    self._P_hamil[bra]     = {}
                self._P_hamil[bra][ket] = deepcopy(hamil_sect)
            if _Z_hamil:
                if not bra in self._Z_hamil:
                    self._Z_hamil[bra]     = {}
                self._Z_hamil[bra][ket] = deepcopy(hamil_sect)
        
        if _dd_hamil:
            _SET_qqnn = set(self._ph_sort_pn_exch_F + self._ph_sort_exch)
            for bra, ket in _SET_qqnn:
                Jmin,  Jmax,  parOK  = self._getJRangeForBraKet(bra, ket)
                if not parOK: continue
                if not bra in self._dd_hamil[3]: self._dd_hamil[3][bra] = {}
                
                hamil_sect = dict([(J, deepcopy(zeros_))for J in range(Jmin, Jmax+1)])
                self._dd_hamil[3][bra][ket] = deepcopy(hamil_sect)
        
        if _P_hamil:
            _SET_qqnn = set(self._ph_sort_pn_exch_P + self._ph_sort_exch_P)
            for bra, ket in _SET_qqnn:
                Jmin,  Jmax,  parOK  = self._getJRangeForBraKet(bra, ket)
                if not parOK: continue
                if not bra in self._P_hamil: self._P_hamil[bra] = {}
                
                hamil_sect = dict([(J, deepcopy(zeros_))for J in range(Jmin, Jmax+1)])
                self._P_hamil[bra][ket] = deepcopy(hamil_sect)
        #
    
    def _setUpHamiltoniannStatesAndAngularFunctions(self):
        """
        Set a default 0 matrix element for all matrix elements in the hamiltonians:
        
        Clebsh-Gordan coefficients that appear in many methods.
        """
        self._resetInternalHamiltonians(_dd_hamil=True, _P_hamil=True, _Z_hamil=True)
        
        ## Set Up the Clebsh-Gordan Coefficients
        _aux_dict = {-1:{}, 0: {}, 1:{},}
        self.__CGC_jab_byJ = dict([(J, deepcopy(_aux_dict)) 
                                        for J in range(self.j_max +1)])
        
        for J in range(self.j_max +1):
            for ja in range(1, self.j_max+1, 2):
                for jb in range(1, self.j_max+1, 2):
                    if abs(ja - jb) > 2*J or (ja + jb) < 2*J: 
                        self.__CGC_jab_byJ[J][0][(ja, jb)] = 0
                        self.__CGC_jab_byJ[J][0][(jb, ja)] = 0
                        self.__CGC_jab_byJ[J][1][(ja, jb)] = 0
                        self.__CGC_jab_byJ[J][1][(jb, ja)] = 0
                        continue
                    aux = [safe_clebsch_gordan(ja/2, jb/2, J, -1/2, -1/2, -1),
                           safe_clebsch_gordan(ja/2, jb/2, J,  1/2, -1/2,  0),
                           safe_clebsch_gordan(ja/2, jb/2, J,  1/2,  1/2, +1), ]
                    self.__CGC_jab_byJ[J][-1][(ja, jb)] = aux[0]
                    self.__CGC_jab_byJ[J][ 0][(ja, jb)] = aux[1]
                    self.__CGC_jab_byJ[J][ 1][(ja, jb)] = aux[2]
    
    @staticmethod
    def pandya_j_scheme_transformation(aa, bb, cc, dd, J, t, Jvals_hamil,
                                       full_hamiltonian=True):
        """
        Transformation to change particle-hole to particle-particle matrix elements.
        (reversible).
            :aa, bb ... = (a-shell index, ja), ...
            :J
            :t indexes for accessing the J-value hamil
            :Jvals_hamil: [(a,b)][(c,d)][J][t] matrix elements in PH format to
                    use in the transformation
        
                <a(b^-1)| v |c(d^-1)>_J = - sum_J2 {6j} <ad| v | cb>_J2
            :return h(ph) ab,cd
        ** NOTE: particle-particle element is equivalent, due the switch b<->d
         
        Considers the switch of particle labeling
            
                PP scheme    PH scheme (remain the type or change to other new)
            0    pp-pp        pp-pp    (0)
            1    pn-pn        pn-pn    (1)
            2    pn-np        pp-nn    *
            3    np-pn        nn-pp    *
            4    np-np        np-np    (4)
            5    nn-nn        nn-nn    (5)
        """
        self = RearrangementTBMESpherical_Runner
        
        a, ja = aa
        b, jb = bb
        c, jc = cc
        d, jd = dd
        Jmin, Jmax, parityOK = self._getJRangeForBraKet((a, d), (c, b))
        if not parityOK: return 0
        convHamil_abcd = 0
        for J2 in range(Jmin, Jmax +1):
            aux  = 0
            if full_hamiltonian:
                if (a, d) in Jvals_hamil:
                    if (c, b) in Jvals_hamil[(a, d)]:
                        if J2 in Jvals_hamil[(a, d)][(c, b)]:
                            aux = Jvals_hamil[(a, d)][(c, b)][J2][t] ## p-h me
            else:
                if J2 in Jvals_hamil: aux = Jvals_hamil[J2][t] ## p-h me
            if almostEqual(aux, 0): continue
            aux *= safe_wigner_6j(ja/2, jb/2, J, jc/2, jd/2, J2) * (2*J2 + 1)
            
            convHamil_abcd += aux
            
        return (-1) * convHamil_abcd
    
    @staticmethod
    def _convertPPNNtoPNPN_phme(aa, bb, cc, dd, J, t, Jvals_hamil,
                                full_hamiltonian=True):
        """
        The method transforms pp-nn / nn-pp particle-hole matrix elements into
        pn-pn / np-np particle-hole matrix elements using the transformation
        _Suhonen_ (9.26)
        
        <p1(n1^-1)| v |p2(n2^-1)>_J = sum_J2 {6j} <p1(p2^-1)| v | n1(n2^-1)>_J2
         a  b          c  d                        a  c           b  d
        
            :aa, bb ... = (a-shell index, ja), ...
            :J
            :t indexes for accessing the J-value hamil
            :Jvals_hamil: [(a,b)][(c,d)][J][t] matrix elements in PH format to
                    use in the transformation
            
            :full_hamiltonian=True: if False, Jvals_hamil=Jvals_hamil[(a, c)][(b, d)]
            
            # NOTE: full Hamiltonian_ must have the pp-nn (t=2) for the pn-pn
            #     (and t=3 for np-np) already calculated.
            
        """
        if not t in (1, 4): return 0
        ## The matrix elements from the hamiltonian J-scheme for pn-pn depend on
        ## 
        #t2 = 2 if (t == 1) else 3
        t2 = t
        
        self = RearrangementTBMESpherical_Runner
        
        a, ja = aa
        b, jb = bb
        c, jc = cc
        d, jd = dd
        Jmin, Jmax, parityOK = self._getJRangeForBraKet((a, c), (b, d))
        if not parityOK: return 0
        convHamil = 0
        for J2 in range(Jmin, Jmax +1):
            aux  = 0
            if full_hamiltonian:
                if (a, c) in Jvals_hamil:
                    if (b, d) in  Jvals_hamil[(a, c)]:
                        if J2 in  Jvals_hamil[(a, c)][(b, d)]:
                            aux = Jvals_hamil[(a, c)][(b, d)][J2][t2]
            else:
                if J2 in Jvals_hamil: aux = Jvals_hamil[J2][t2]
            
            if almostEqual(aux, 0): continue
            aux *= safe_wigner_6j(ja/2, jb/2, J, jd/2, jc/2, J2) * (2*J2 + 1)
            aux *= (-1)**J2
            
            convHamil += aux
        return (-1)**(jb/2 + jc/2 + J + 1) * convHamil
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """
        Defined as for a normal interaction SHO and Dens-Dep. parameters.
        """
        for k, vals in kwargs.items():
            if k == _DD_pe.file:
                cls.filename = vals['name']
            else:
                if isinstance(vals, dict):
                    if   k == _DD_pe.alpha:
                        cls.ALPHA    = vals['value']
                    elif k == _DD_pe.constant:
                        cls.CONST_t0 = vals['value']
                    else:
                        setattr(cls, k, vals['value'])
                else:
                    ## hbar omega & b
                    if k == SHO_Parameters.b_length:
                        cls.B_LEN = vals
                    else:
                        setattr(cls, k, vals)
    
    @classmethod
    def _readWaveFunctionFromFile(cls):
        """ 
        Get the file from U, V generated by Taurus. 
            # number of shell states
            1
            2 ... shell states
            hash label state (ignored)
            U[j,i] values
            V[j,i] values
        
        Required conversion to the density matrix    rho = V^* V^T
        """
        sh_dim = 0
        density_matrix_lines = []
        with open(cls.filename, 'r') as f:
            data = f.readlines()
            for il, line in enumerate(data):
                line = line.strip()
                if il == 0:
                    sh_dim = int(line)
                elif il <= sh_dim:
                    sh_st = int(line)
                    cls.sh_states.append(sh_st)
                    n, l, j = readAntoine(sh_st, l_ge_10=True)
                    
                    cls.j_max = max(j, cls.j_max)
                    cls.l_max = max(l, cls.l_max)
                    cls.n_max = max(n, cls.n_max)
                    
                    cls.sh_states_obj[sh_st] = QN_1body_jj(n, l, j)
                elif il == sh_dim + 1:
                    continue
                else:
                    # if 'E' in line:
                    #     line = line.replace('E', 'e')
                    line = float(line)
                    density_matrix_lines.append(line)
        
        cls.sh_dim = sh_dim
        
        dim_ = len(density_matrix_lines) // 2
        u0_mat = density_matrix_lines[:dim_]
        v0_mat = density_matrix_lines[dim_:]
        
        ## NOTE: verified with the DIMENS_indexes_and_rhoLRkappas.gut file
        cls._setUpDensityMatrices(dim_, u0_mat, v0_mat)
        
        ## define the ab, cd in order for the
        for i, a_sh in enumerate(cls.sh_states):
            for b_sh in cls.sh_states[i:]:
                cls.sh_states_sorted.append( (a_sh, b_sh) )                
        
        # import matplotlib.pyplot as plt
        plt.imshow(cls.U_sp)
        plt.show()
        plt.imshow(cls.V_sp)
        plt.show()
        plt.imshow(np.array(cls._density_matrix))
        plt.show()
        plt.imshow(np.array(cls._kappa_matrix))
        plt.show()
    
    @classmethod
    def _setUpDensityMatrices(cls, dim_, u0_mat, v0_mat):
        """ Auxiliary to set the density matrices shared when read
            (once processed from text or binary file)"""
        dim_ = int (dim_**0.5)
        
        U, V = np.zeros((dim_, dim_)), np.zeros((dim_, dim_))
        cls._density_matrix = np.zeros((dim_, dim_))
        cls._kappa_matrix   = np.zeros((dim_, dim_)) 
        for i in range(dim_):
            for j in range(dim_):
                k = i*dim_ + j
                U[j, i] = u0_mat[k]
                V[j, i] = v0_mat[k]
        
        cls.sp_dim = dim_
        cls.U_sp = U
        cls.V_sp = V
        cls._density_matrix = np.matmul(V.conjugate(), np.transpose(V)) 
        cls._kappa_matrix   = np.matmul(V.conjugate(), np.transpose(U))
        
        cls.A = np.trace(cls._density_matrix)
        cls.Z = np.trace(cls._density_matrix[dim_//2:, dim_//2:])
        cls.N = cls.A - cls.Z
        
        ## Set the sp_states in the file order
        k = 0
        for mt in (1, -1):
            for sh_st in cls.sh_states:
                sh_obj = cls.sh_states_obj[sh_st]
                n, l, j = sh_obj.n, sh_obj.l, sh_obj.j
                m_vals = [i for i in range(j, -j-1, -2)]
                for m in m_vals:
                    cls.sp_states.append( QN_1body_jj(n, l, j, m, mt) )
        
        ## VERIFY THE SPHERICAL WAVE FUNCTION AND SET UP THE sh UV matrices.
        ## from cls._isSphericalWF()
        
        is_spherical = True
        for a, st_a in enumerate(cls.sp_states):
            for b in range(a, len(cls.sp_states)):
                st_b = cls.sp_states[b]
                
                if st_a.m_t != st_b.m_t: continue
                if not almostEqual(cls._density_matrix[b, a], 0.0, 1.0e-7):
                    is_spherical *= st_a.m == st_b.m
        
        if not bool(is_spherical):
            raise MatrixElementException("The wave function is NOT spherical.")
        else:
            print(" [TEST] Imported wf is spherically symmetric.")
        
        def __f(x):
            return x # abs(x) #
        
        k, spO2 = 0, cls.sp_dim // 2
        for a, st_a in enumerate(cls.sh_states):
            cls.occupied_states[ 1][st_a] = False
            cls.occupied_states[-1][st_a] = False
        
        for a, st_a in enumerate(cls.sh_states):
            sh_o = cls.sh_states_obj[st_a]
            
            cls.U_sh[ 1][st_a] = __f(U[k, k])
            cls.U_sh[-1][st_a] = __f(U[k+spO2, k+spO2])
            
            k2 = k + sh_o.j
            cls.V_sh[ 1][st_a] = __f(V[k, k2])
            cls.V_sh[-1][st_a] = __f(V[k+spO2, k2+spO2])
            
            if abs(cls.V_sh[ 1][st_a]) > cls.OCCUPIED_TOL:
                cls.occupied_states[ 1][st_a] = True
            if abs(cls.V_sh[-1][st_a]) > cls.OCCUPIED_TOL:
                cls.occupied_states[-1][st_a] = True
            k += sh_o.j + 1
        
    @classmethod
    def setIntegrationGrid(cls):
        """
        Reset the grid for integration for Laguerre-Associated (radial) 
        and Legendre (Angular).
            Implement Levedev Integral (At least one for a large Omega order)
        """
        if SCIPY_INSTALLED:
            from scipy.special import roots_legendre
            from scipy.special import roots_genlaguerre
        else:
            raise MatrixElementException("Scipy Required to use this MatrixElement")
        
        ## Radial Grid
        ## From method _Base_DensityDependent.setRadialIntegrationGrid  =======
        xR, wR, sum_ = roots_genlaguerre(cls._R_DIM, 0.5, True)
        cls._r =  cls.B_LEN * np.sqrt(xR / (2.0 + cls.ALPHA)) # 
        cls._weight_r = wR
        ## ====================================================================
        
        ## Angular Grid: Choice between stored Lebedev_ grid mesh (commonly used).
        cls._ang = []
        cls._weight_ang = []
        if cls.USING_LEBEDEV:
            ## Angular Grid from Lebedev_ Imported from /docs/LebedevPointsWeights
            if cls._OMEGA % 2 == 1:
                print(" WARNING: Valid Omega Orders [14,16,18,20,22], OMEGA +=1")
                cls._OMEGA += 1
            if cls._OMEGA < 14:
                print(" WARNING: Minimum Omega for Lebedev is 14, fixing to that.")
                cls._OMEGA = 14
            elif cls._OMEGA > 22:
                print(" WARNING: Maximum Omega for Lebedev is 22, fixing to that.")
                cls._OMEGA = 22
            ## read the file and convert the numbers
            f_ = f'docs/LebedevPointsWeightsOmg{cls._OMEGA}.gut'
            if not os.path.exists(f_): f_ = '../' + f_
            cls._A_DIM = 0
            with open(f_, 'r') as f:
                for line in f.readlines()[1:]:
                    line = line.strip().replace('D', 'E')
                    i, costh, phi, wA = line.split()
                    cls._ang.append( (float(costh), float(phi)) )
                    cls._weight_ang.append( float(wA) )
                    cls._A_DIM += 1
        else:
            ## Angular Grid (Theta_dim=Phi_dim = Ang_dim)
            xA, wA, sum_ = roots_legendre(cls._OMEGA, True)
            costh = xA
            phi   = np.pi * copy(xA)
            for i in range(cls._OMEGA):
                for j in range(cls._OMEGA):
                    cls._ang.append( (costh[i], phi[j]) )
                    cls._weight_ang.append(wA[i] * wA[j])
            cls._A_DIM = cls._OMEGA**2
        
        cls._spatial_dens        = np.zeros( (cls._R_DIM, cls._A_DIM) )
        cls._spatial_dens_pn     = [np.zeros( (cls._R_DIM, cls._A_DIM) ),
                                    np.zeros( (cls._R_DIM, cls._A_DIM) )]
        cls._spatial_dens_imag   = np.zeros( (cls._R_DIM, cls._A_DIM) ) ## test
        cls._spatial_dens_alp    = np.zeros( (cls._R_DIM, cls._A_DIM) )
        cls._spatial_dens_alpM1  = np.zeros( (cls._R_DIM, cls._A_DIM) )
        cls._spatial_dens_alpM2  = np.zeros( (cls._R_DIM, cls._A_DIM) )  
    
    @classmethod
    def _setUpSHOBasisRadialAngular(cls):
        """
        Once given the w.f. and having the b_lengths (core, m.e.)
        Set the angular and Radial functions and angular coefficents.
        """
        cls.setIntegrationGrid()
        
        radial2b = _buildRadialFunctionsArray(cls.sh_states, cls.B_LEN)
        cls._radial_2b_wf_memo = deepcopy(radial2b)
        
        # Build Base for Angular and Radial coefficients of the quantum numbers
        K_max = _buildAngularYCoeffsArray(cls.sh_states,
                                          reset_file_test = True)
        cls._orbital_max_sph_harmonic = K_max
        
        # from scipy.special import sph_harm
        ## Angular basis (Common for both cases)       
        for K in range(cls._orbital_max_sph_harmonic +1):
            for M in range(-K, K +1):
                ind_k = angular_Y_KM_index(K, M, False)
                
                if ind_k in cls._sph_harmonic_memo: continue
                
                ang_func = [sphericalHarmonic(K, M, angle) for angle in cls._ang]
                ang_func = np.array(ang_func)
                cls._sph_harmonic_memo[ind_k] = deepcopy(ang_func)
        
    @classmethod
    def _setUpDensities(cls):
        """
        Evaluate the radial and angular functions for the Mean field W.F.
        """
        global _angular_Y_KM_me_memo
        t_0 = time.time()
        print(f" [ ] Evaluating Spatial Densities ...")
        
        for a  in range(cls.sp_dim // 2):
            # print("  progr. a% =", tot_*(1 - a) - (a*(a-1)//2), "/", tot_)
            if a % 5 == 0: print(f"  progr a:{a+1:>3.0f}/{cls.sp_dim//2:>3.0f}")
            sp_st_a = cls.sp_states[a]
            na, la, ja, mja = sp_st_a.n, sp_st_a.l, sp_st_a.j, sp_st_a.m
            indx_a = angular_Y_KM_index(ja, mja, True)
            
            for b in range(a, cls.sp_dim // 2):
                sp_st_b = cls.sp_states[b]
                
                if sp_st_a.m_t != sp_st_b.m_t :
                    continue
                nb, lb, jb, mjb = sp_st_b.n, sp_st_b.l, sp_st_b.j, sp_st_b.m
                indx_b = angular_Y_KM_index(jb, mjb, True)
                                
                rad_key = (na,la, nb,lb)
                cls._aux_radial  = cls._radial_2b_wf_memo[rad_key](cls._r)
                
                cls._angularComponentsForDensity(a,la,ja,mja,indx_a, 
                                                 b,lb,jb,mjb,indx_b)
                if a != b:
                    cls._angularComponentsForDensity(b,lb,jb,mjb,indx_b,
                                                     a,la,ja,mja,indx_a)
        
        cls._spatial_dens_alp   = np.power(cls._spatial_dens, cls.ALPHA)
        cls._spatial_dens_alpM1 = np.power(cls._spatial_dens, cls.ALPHA - 1)
        cls._spatial_dens_alpM2 = np.power(cls._spatial_dens, cls.ALPHA - 2)
        ## TODO: verify the CUToff for the alpha-1 -2 conditions.
        
        cls._testDensityIntegratingForMassNumber(t_0)
        
        ## check the profile of radial proton / neutron densities
        # pp, nn = [], []
        # for ir in range(cls._R_DIM):
        #     xp = np.inner(cls._weight_ang, cls._spatial_dens_pn[0][ir, :])
        #     pp.append(xp)
        #     xn = np.inner(cls._weight_ang, cls._spatial_dens_pn[1][ir, :])
        #     nn.append(xn)
        # plt.plot(cls._r, pp, 'r-', cls._r, nn, 'b-')
        # plt.show()
    
    @classmethod
    def _angularComponentsForDensity(cls, a,la,ja,mja,indx_a, b,lb,jb,mjb,indx_b):
        """ Find the K,M components of the two body angular wave function and
        add up to the spatial density 
        
        Note, in order to speed up the density matrix summation"""
        spO2 = cls.sp_dim // 2
        dens_pn = cls._density_matrix[b, a], cls._density_matrix[b+spO2, a+spO2]
        
        if sum(dens_pn) < 1.0e-6: return 
        
        for K in range(abs(ja-jb)//2, (ja+jb)//2 +1):
            M = (mjb - mja) // 2
            if ((abs(M) > K) or ((K + la + lb) % 2 == 1)):
                continue
            
            indx_K  = angular_Y_KM_index( K,   M, False)
            key_abK = _angular_Y_KM_memo_accessor(indx_a,indx_b,indx_K)
            
            c_ang   = _angular_Y_KM_me_memo[key_abK]
            
            for ir in range(len(cls._r)):
                for ia in range(len(cls._ang)):
                    
                    angular = cls._sph_harmonic_memo[indx_K][ia] * c_ang
                    val = cls._aux_radial[ir] * angular
                    
                    val_p = val * dens_pn[0]
                    val_n = val * dens_pn[1]
                    
                    val = val_p + val_n
                    cls._spatial_dens_imag[ir, ia] += np.imag(val)
                    cls._spatial_dens[ir, ia]      += np.real(val)
                    
                    cls._spatial_dens_pn[0][ir, ia] += np.real(val_p)
                    cls._spatial_dens_pn[1][ir, ia] += np.real(val_n)
    
    @classmethod
    def _testDensityIntegratingForMassNumber(cls, t_start):
        """
        This test verify int(density) = A
        """
        integ_A = 0.0
        for ir in range(len(cls._r)):
            radial  = np.exp( ( (2.0 + cls.ALPHA) * 
                                (cls._r[ir] / cls.B_LEN)**2) )
            for ia in range(len(cls._ang)):
                val  = cls._spatial_dens[ir, ia] 
                val *= cls._weight_ang[ia] * cls._weight_r[ir]
                integ_A +=  radial * val
        
        integ_A *= (cls.B_LEN**3) / (2.0*((2.0 + cls.ALPHA)**1.5))
        if cls.USING_LEBEDEV:
            integ_A *= 4 * np.pi
        else:
            integ_A *= np.pi ## not checked
        
        t_ = (time.time() - t_start)
        print( " [DONE] Spatial Density has been imported and evaluated. ",
              f"({t_:5.3f}s) A={integ_A:9.5f}")
    
    
    def _calculateDDMatrixElements_2bme(self):
        """
        Class dedicated to the integrals involved in Matrix element
        
            PP matrix elements    <ab|vDD|cd>_J
        
        Evaluating for p-p interaction, as the normal density-dependent Hamiltoninan
        
        Not to include in the p-h -> p-p transformation.
        """
        self._ME_TYPE = 0
        
        ## When pnpn_case:     ad_sh & cb_sh   -->  ac_sh, db_sh
        for i, states in enumerate(self._2b_sort_states):
            ab_sh, cd_sh = states
            ad_sh, cb_sh = self._ph_sort_states[i]
            
            if (*ab_sh, *cd_sh) == (101, 203, 103, 10001):
                _ = 0
            # TODO: J range
            Jmin, Jmax, parityOK = self._getJRangeForBraKet(ab_sh, cd_sh)
            
            if not parityOK: continue
            bra01, bra02 = readAntoine(ab_sh[0], 1), readAntoine(ab_sh[1], 1)
            ket01, ket02 = readAntoine(cd_sh[0], 1), readAntoine(cd_sh[1], 1)
            
            print("*** Eval for: {}{} J-range={}:{}  ================"
                  .format(ab_sh, cd_sh, Jmin, Jmax))
            # Element DD
            for J in range(Jmin, Jmax +1):
                for t in range(6):
                    self.t_indx = t
                    tt = TBME_Reader._JSchemeIndexing[t]
                    
                    ## Only evaluate the particle-hole orbits (occupied vs empty)
                    if self.APPLY_PH_OCCUPATION_DD_2BME:
                        skip = False
                        if   (t in (0, 2)):
                            t1 =  1 if t== 0  else -1
                            _ph_b = [self.occupied_states[ 1][x] for x in ad_sh]
                            _ph_k = [self.occupied_states[t1][x] for x in cb_sh]
                            skip = (_ph_b[0]!=_ph_b[1]) and (_ph_k[0]!=_ph_k[1])
                        elif (t in (3, 5)):
                            t1 =  1 if t== 3  else -1
                            _ph_b = [self.occupied_states[-1][x] for x in ad_sh]
                            _ph_k = [self.occupied_states[t1][x] for x in cb_sh]
                            skip = (_ph_b[0]!=_ph_b[1]) and (_ph_k[0]!=_ph_k[1])
                        else:
                            t1 =  1 if t== 1  else -1
                            _ph = [
                                self.occupied_states[ t1][ad_sh[0]],
                                self.occupied_states[-t1][ad_sh[1]],
                                self.occupied_states[ t1][cb_sh[0]],
                                self.occupied_states[-t1][cb_sh[1]]  ]
                            skip = (_ph[0]!=_ph[1]) and (_ph[2]!=_ph[3])
                        if skip: continue
                    
                    v_dd   = 0
                    self.J = J
                    bra0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*bra01, mt= tt[0]),
                                                    QN_1body_jj(*bra02, mt= tt[1]), J)
                    ket0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*ket01, mt= tt[2]),
                                                    QN_1body_jj(*ket02, mt= tt[3]), J)
                    self.bra = bra0wf
                    self.ket = ket0wf
                    
                    norm_ = self.bra.norm() * self.ket.norm()
                    if norm_ != 0:
                        v_dd = self._antisymmetrized_J_element() * norm_
                    
                    if abs(v_dd) > self.NULL_TOLERANCE: 
                        self._saveDDMatrixElement(v_dd, ab_sh, cd_sh)
                        print("    * v0: {}{} J={} = {:7.3e}"
                              .format(ab_sh, cd_sh, J, v_dd)) #############
                print()
        
    
    def _calculateDDMatrixElements(self):
        """
        Class dedicated to the integrals involved in Matrix element
        
            PH matrix elements    <ab|vDD|cd>_J
            
        ## Note, PH matrix element for V-dd is not mentioned in the article, 
        we perform the Pandya transformation, where it is easy to obtain both
        qq,q'q' and pn-pn matrix elements:
                                            
        <a d^-1|v|c b^-1>_J(t) = - sum_J' ... (6j case) <ab | v | cd>_J'(t)
        
            1. Evaluate the (J to J') <ab|v|cd> (t) - matrix elements
            2. Copy into the <ad|cb>-t PH
        
        """
        self._ME_TYPE = 0
        
        for i,  states in enumerate(self._2b_sort_states):
            ab_sh, cd_sh = states
            ad_sh, cb_sh = self._ph_sort_states[i]
            
            # TODO: J range
            Jmin, Jmax, parityOK  = self._getJRangeForBraKet(ad_sh, cb_sh)
            Kmin, Kmax, parity2OK = self._getJRangeForBraKet(ab_sh, cd_sh)
            
            if not parityOK: continue
            bra01, bra02 = readAntoine(ab_sh[0], 1), readAntoine(ab_sh[1], 1)
            ket01, ket02 = readAntoine(cd_sh[0], 1), readAntoine(cd_sh[1], 1)
            
            print("*** Eval DD for: {}{} J-range={}:{}  ================"
                  .format(ad_sh, cb_sh, Jmin, Jmax))
            # Element DD
            for J in range(Jmin, Jmax +1):
                for t in range(6):
                    self.t_indx = t
                    tt = TBME_Reader._JSchemeIndexing[t]
                    
                    ## Only evaluate the particle-hole orbits (occupied vs empty)
                    if self.APPLY_PH_OCCUPATION_DD_2BME:
                        skip = False
                        if   (t in (0, 2)):
                            t1 =  1 if t== 0  else -1
                            _ph_b = [self.occupied_states[ 1][x] for x in ad_sh]
                            _ph_k = [self.occupied_states[t1][x] for x in cb_sh]
                            skip = (_ph_b[0]==_ph_b[1]) or (_ph_k[0]==_ph_k[1])
                        elif (t in (3, 5)):
                            t1 =  1 if t== 3  else -1
                            _ph_b = [self.occupied_states[-1][x] for x in ad_sh]
                            _ph_k = [self.occupied_states[t1][x] for x in cb_sh]
                            skip = (_ph_b[0]==_ph_b[1]) or (_ph_k[0]==_ph_k[1])
                        else:
                            t1 =  1 if t== 1  else -1
                            _ph = [
                                self.occupied_states[ t1][ad_sh[0]],
                                self.occupied_states[-t1][ad_sh[1]],
                                self.occupied_states[ t1][cb_sh[0]],
                                self.occupied_states[-t1][cb_sh[1]]  ]
                            skip = (_ph[0]==_ph[1]) or (_ph[2]==_ph[3])
                        if skip: continue
                    
                    v_dd   = 0
                    self.J = J
                                        
                    for K in range(Kmin, Kmax +1):
                        
                        args = [bra01[2]/2, ket02[2]/2, J, 
                                ket01[2]/2, bra02[2]/2, K,] 
                            
                        fac6j = safe_wigner_6j(*args)
                        if self.isNullValue(fac6j): continue
                        
                        factor = (2*K + 1) * fac6j
                        
                        bra0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*bra01, mt= tt[0]),
                                                        QN_1body_jj(*bra02, mt= tt[1]), 
                                                        K)
                        ket0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*ket01, mt= tt[2]),
                                                        QN_1body_jj(*ket02, mt= tt[3]), 
                                                        K)
                        self.bra = bra0wf
                        self.ket = ket0wf
                        
                        norm_ = self.bra.norm() * self.ket.norm()
                        if norm_ != 0:
                            v_dd += self._antisymmetrized_J_element() * norm_ * factor
                    
                    v_dd *= -1
                    if abs(v_dd) > self.NULL_TOLERANCE:
                        self._saveDDMatrixElement(v_dd, ad_sh, cb_sh)
                        print("    * v0: {}{} J={} = {:7.3e}"
                              .format(ad_sh, cb_sh, J, v_dd)) #############
                print()
              
    
    def _calculateDDhamilForFparticleHole(self):
        """
        Obtain the matrix elements for F m.e. for the qq,qq and qq,q'q'
        
        NOTE: particle label for the ph matrix elements relate to the pp ones,
        being 2:ppnn (from pnnp) and 3:nnpp (from the nppn), the rest match.
        """
        self._ME_TYPE = 3
        for i, states in enumerate(self._ph_sort_states):
            ad_sh, cb_sh   = states
            self.__sh_curr = states
            
            cb_sh_exch = (cb_sh[1], cb_sh[0])
            _eval_exch = cb_sh[0] != cb_sh[1]
            
            bra01, bra02 = readAntoine(ad_sh[0], 1), readAntoine(ad_sh[1], 1)
            ket01, ket02 = readAntoine(cb_sh[0], 1), readAntoine(cb_sh[1], 1)
            
            Jmin, Jmax, parityOK = self._getJRangeForBraKet(ad_sh, cb_sh)
            
            if not parityOK: continue
            
            for J in range(Jmin, Jmax +1):
                for t in (0, 2, 3, 5):
                    self.t_indx = t
                    
                    ## Not necessary the EXCH option, 2 & 3 remain pp-nn & nn-pp
                    ta,tb,tc,td = TBME_Reader._JSchemeIndexing[t]
                    
                    self.J = J
                    bra0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*bra01, mt= ta),
                                                    QN_1body_jj(*bra02, mt= td), J)
                    ket0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*ket01, mt= tc),
                                                    QN_1body_jj(*ket02, mt= tb), J)
                    self.bra = bra0wf
                    self.ket = ket0wf
                    
                    norm_ = self.bra.norm() * self.ket.norm()
                    v_dd_dir, v_dd_exc = [0, 0, 0], [0, 0, 0]
                    ## Eval. F(abcd|J) and F(abdc|J)
                    if norm_ != 0:
                        if self._F_REA_2EVAL[0]:
                            v_dd_dir[0] = self._FRea0_matrixElement()
                        if self._F_REA_2EVAL[1]:
                            v_dd_dir[1] = self._FRea1_matrixElement()
                        if self._F_REA_2EVAL[2]:
                            v_dd_dir[2] = self._FRea3_matrixElement()
                        
                        if _eval_exch:
                            if self._F_REA_2EVAL[0]:
                                v_dd_exc[0] = self._FRea0_matrixElement(True)
                            if self._F_REA_2EVAL[1]:
                                v_dd_exc[1] = self._FRea1_matrixElement(True)
                            if self._F_REA_2EVAL[2]:
                                v_dd_exc[2] = self._FRea3_matrixElement(True)
                        
                    if t in (0, 5): ## F_1_adcb = F_1_adcb q=q'
                        v_dd_dir[1] *= 2
                        v_dd_exc[1] *= 2
                    
                    for i in range(3): 
                        v_dd_dir[i] *= norm_
                        v_dd_exc[i] *= norm_
                    
                    _x = [abs(x) for x in (*v_dd_dir, *v_dd_exc)]
                    if sum(_x) > self.NULL_TOLERANCE:
                        self._saveDDMatrixElement(sum(v_dd_dir), ad_sh, cb_sh)
                        if _eval_exch:
                            self._saveDDMatrixElement(sum(v_dd_exc), ad_sh, cb_sh_exch)
                        _x = [f"{x:7.3e}" for x in (*v_dd_dir, *v_dd_exc)]
                        print("    * F v1 t[{}]: {}{} J={} = {} (exc) {}"
                              .format(t, ad_sh, cb_sh, J, '  '.join(_x[:3]), 
                                      '  '.join(_x[3:])))        #############
                print()
        ## Do after the pn-pn
        # self._exportHamilsDDTesting()
            
    def _calculateDDhamilForFparticleHole_PNPN(self):
        """
        Obtain the matrix elements for F m.e. for the pn-pn and np-np terms, 
        
        Note, calculates the pp-nn and nn-pp to be converted into t=1, 4 PH m.e.
        
        """
        self._ME_TYPE = 3
        for EXCH_ in (0, 1):
            
            list_to_read = self._ph_sort_pn_exch_F if EXCH_ else self._ph_sort_pn_dir
            
            for i, states in enumerate(list_to_read):
                bra_sh, ket_sh = states  
                ## (ac, db) if DIRECT, else (ab, dc) if EXCH
                self.__sh_curr = states
                
                bra01, bra02 = readAntoine(bra_sh[0], 1), readAntoine(bra_sh[1], 1)
                ket01, ket02 = readAntoine(ket_sh[0], 1), readAntoine(ket_sh[1], 1)
                
                Jmin, Jmax, parityOK = self._getJRangeForBraKet(bra_sh, ket_sh)
                
                if not parityOK: continue
                
                for J in range(Jmin, Jmax +1):
                    for t in (1, 4):
                    
                        self.t_indx = t
                        
                        ## Exch_ states given, fix pp-nn & nn-pp
                        ta,tb,tc,td = TBME_Reader._JSchemeIndexing[t]
                        if EXCH_ == 0: t1, t2 = tc, tb
                        else:          t1, t2 = tb, tc
                        
                        self.J = J
                        bra0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*bra01, mt= ta),
                                                        QN_1body_jj(*bra02, mt= t1), J)
                        ket0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*ket01, mt= td),
                                                        QN_1body_jj(*ket02, mt= t2), J)
                        self.bra = bra0wf
                        self.ket = ket0wf
                        
                        norm_ = self.bra.norm() * self.ket.norm()
                        v_dd = [0, 0, 0]
                        ## Eval. F(abcd|J) and F(abdc|J)
                        if norm_ != 0:
                            if self._F_REA_2EVAL[0]:
                                v_dd[0] = self._FRea0_matrixElement()
                            if self._F_REA_2EVAL[1]:
                                v_dd[1] = self._FRea1_matrixElement()
                            if self._F_REA_2EVAL[2]:
                                v_dd[2] = self._FRea3_matrixElement()
                        
                        for i in range(3): 
                            v_dd[i] *= norm_
                        
                        _x = [abs(x) for x in v_dd]
                        if sum(_x) > self.NULL_TOLERANCE:
                            self._saveDDMatrixElement(sum(v_dd), bra_sh, ket_sh)
                            _x = [f"{x:7.3e}" for x in v_dd]
                            print("    * F v1 t[{}]: {}{} J={} = {} (exc) {}"
                                  .format(t, bra_sh, ket_sh, J, '  '.join(_x[:3]), 
                                          '  '.join(_x[3:]))) #############
                print()
        ## NOTE: Do it after the pp-nn to pp-nn conversion
        #self._exportHamilsDDTesting()
    
    
    def _calculateDDMatrixElements_2ndpart(self):
        """
        Matrix elements associated to the first derivative of rho(r)
        
            C_abcd * Sum_e {C_ef UV_ef} *  <ff|d vDD phi(ab)phi(cd)|ee>_J=0>
        """
        self._ME_TYPE = 1
        self.J = 0
        ## pp-nn 
        for states in self._ph_sort_states:
            ad_sh, cb_sh = states
            for t in (0, 2, 3, 5): ## states pp-pp & nn-nn
                self.t_indx = t
                self._Pi_abcd_matrix_general(ad_sh, cb_sh, 0)
        
                ## pp-nn EXCHANGED
                ## If the element repeats, do not sum again.
                if (cb_sh == ad_sh): 
                    continue
                if (ad_sh[0] == ad_sh[1]) or (cb_sh[0] == cb_sh[1]):
                    continue
                self._Pi_abcd_matrix_general(cb_sh, ad_sh, 1)
        
        ## pn-pn
        for states in self._ph_sort_pn_dir:
            ac_sh, db_sh = states
            for t in (1, 4): ## states pp-pp & nn-nn
                self.t_indx = t
                self._Pi_abcd_matrix_general(ac_sh, db_sh, 2)
        
        ## pn-pn EXCHANGED
        for states in self._ph_sort_pn_exch_P:
            ca_sh, bd_sh = states
            for t in (1, 4): ## states pp-pp & nn-nn
                self.t_indx = t
                ## If the element repeats, do not sum again.
                if (ca_sh[0] == ca_sh[1]) or (bd_sh[0] == bd_sh[1]):
                    continue
                self._Pi_abcd_matrix_general(ca_sh, bd_sh, 3)
        
    
    def _Pi_abcd_matrix_general(self, ab_sh, cd_sh, _ICASE=0):
        """
        Pi matrix elements is cumbersome to separate into different pp-nn, exchanged
        etc.
        : indicate the case _ICASE: (not relevant)
            _ICASE = 0: dire
            _ICASE = 1: exch
            _ICASE = 2: dire - pn-pn (for pp-nn states)
            _ICASE = 3: exch - pn-pn (for pp-nn states)
        """
        bra01, bra02 = readAntoine(ab_sh[0], 1), readAntoine(ab_sh[1], 1)
        ket01, ket02 = readAntoine(cd_sh[0], 1), readAntoine(cd_sh[1], 1)
        
        ## skipping deltas for abcd:
        _OK = [
            bra01[1] == bra02[1], bra01[2] == bra02[2],
            ket01[1] == ket02[1], ket01[2] == ket02[2],
            self.J == 0,
        ]
        if not all(_OK): return
        
        ja = self.sh_states_obj[ab_sh[0]].j
        jc = self.sh_states_obj[ab_sh[0]].j
        
        ## Factor is common to the ee sum, evaluate outside the radial integral
        factor  = 0.5 * self.CONST_t0 * self.ALPHA * (1 - self.x0)
        factor *= np.sqrt((ja + 1)*(jc + 1))
        factor /= (4 * np.pi)**2
        
        for ee in self.sh_states:
            ee_sh = (ee, ee)
            Jmin, _, parityOK = self._getJRangeForBraKet(ab_sh, ee_sh)
            
            if not parityOK or Jmin > 0: continue
            
            ## J = 0
            ## Not necessary the EXCH option, 2 & 3 remain pp-nn & nn-pp
            ta,tb, tc,td = TBME_Reader._JSchemeIndexing[self.t_indx]
            
            if   self.t_indx in (0, 5):
                tts = (ta,tb, tc,td)
                TT  =  1 if self.t_indx == 0 else -1
                assert _ICASE in (0, 1), "Invalid use of the Pi m.e. (pp-pp)= ICASE 0, 1"
            elif self.t_indx in (2, 3):
                tts = (ta,td, tc,tb) if _ICASE == 0 else (tc,tb, ta,td)
                TT  = ta if _ICASE == 0 else tc
                assert _ICASE in (0, 1), "Invalid use of the Pi m.e. (pp-nn)= ICASE 0, 1"
            elif self.t_indx in (1, 4):
                tts = (ta,tc, td,tb) if _ICASE == 2 else (ta,tc, tb,td)
                TT  = ta
                assert _ICASE in (2, 3), "Invalid use of the Pi m.e. (pn-pn) -> (pp-nn) ICASE 2, 3"
            else:
                raise Exception("WTF t index you set?")
                
            ee01 = readAntoine(ee, 1)
            _, le, je = ee01
            phs = (-1)**(le + bra01[1])
            
            _UVe_fact = (je + 1) * self.U_sh[TT][ee] * self.V_sh[TT][ee] * phs
            
            bra0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*bra01, mt= tts[0]),
                                            QN_1body_jj(*bra02, mt= tts[1]), self.J)
            ket0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*ee01,  mt= TT),
                                            QN_1body_jj(*ee01,  mt= TT), self.J)
            fix0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*ket01, mt= tts[2]),
                                            QN_1body_jj(*ket02, mt= tts[3]), self.J)
            self.bra   = bra0wf
            self.ket   = ket0wf
            self.fix_1 = fix0wf
            self.__sh_curr = (ab_sh, cd_sh, ee)
            
            norm_ = self.bra.norm() *  self.ket.norm()  ## Does not appear in the expressions!!
            v_dd = 0
            if norm_ != 0:
                v_dd = self._radial_integral_PiZeta_calculation(1)  
            v_dd *= _UVe_fact * factor * norm_
            
            ## Save the DD-matrix element with the intermediate value for TESTING
            self._saveDDMatrixElement (v_dd, ab_sh, ee_sh, fix_1 = cd_sh)
            self._saveREAMatrixElement(v_dd, ab_sh, cd_sh)
            
            if abs(v_dd) > self.NULL_TOLERANCE: 
                print("      * v1: {}{}-{} = [t={}][IC:{}] {:7.3e}"
                      .format(ab_sh, cd_sh, ee, self.t_indx, _ICASE, v_dd)) ###
        print()
        
    
    def _calculateDDMatrixElements_3rdpart(self):
        """
        Matrix elements associated to the second derivative of rho(r)
        
            C_abcdJ * Sum_e {C_ef UV_ef} * <ee|dd vDD phi(ab)phi(cd)|ff>_J=0>
        """
        self._ME_TYPE = 2
        ## pp-nn 
        for states in self._ph_sort_states:
            ad_sh, cb_sh = states
            for t in (0, 2, 3, 5): ## states pp-pp & nn-nn
                self.t_indx = t
                self._Zeta_abcd_matrix_general(ad_sh, cb_sh, 0)
        
        ## pn-pn
        for states in self._ph_sort_pn_dir:
            ac_sh, db_sh = states
            for t in (1, 4): ## states pp-pp & nn-nn
                self.t_indx = t
                self._Zeta_abcd_matrix_general(ac_sh, db_sh, 2)
        # Note: No exchange, the matrix elements are in ee-ff (doubled)
    
    def _Zeta_abcd_matrix_general(self, ab_sh, cd_sh, _ICASE=0):
        """
        Pi matrix elements is cumbersome to separate into different pp-nn, exchanged
        etc.
        : indicate the case _ICASE: (not relevant)
            _ICASE = 0: dire
            _ICASE = 2: dire - pn-pn (for pp-nn states)
        """
        bra01, bra02 = readAntoine(ab_sh[0], 1), readAntoine(ab_sh[1], 1)
        ket01, ket02 = readAntoine(cd_sh[0], 1), readAntoine(cd_sh[1], 1)
        _auxJ = self._getJRangeForBraKet(ab_sh, cd_sh) ## Only to avoid bra-Jab exception.
        ## skipping deltas for abcd:
        _OK = [
            (bra01[1] + bra02[1] + self.J) % 2 == 0,
            (ket01[1] + ket02[1] + self.J) % 2 == 0,
        ]
        if not all(_OK): return
        a, b = self.sh_states_obj[ab_sh[0]], self.sh_states_obj[ab_sh[1]]
        c, d = self.sh_states_obj[cd_sh[0]], self.sh_states_obj[cd_sh[1]]
        
        factor  = 0.25 * self.CONST_t0 * self.ALPHA * (self.ALPHA - 1)
        factor *= np.sqrt((a.j+1)*(b.j+1)*(c.j+1)*(d.j+1))
        factor *= (-1)**((a.j - c.j)/2) 
        factor *= self.__CGC_jab_byJ[self.J][0][(a.j, b.j)]
        factor *= self.__CGC_jab_byJ[self.J][0][(c.j, d.j)]
        factor /= (2*self.J + 1) * ((4 * np.pi)**3)
        
        k = {'ignoreCheck': True, }
        for ee in self.sh_states:
            ee_sh = (ee, ee)
            ee01 = readAntoine(ee, 1)
            _, le, je = ee01
            for ff in self.sh_states:
                ff_sh = (ff, ff)
                ff01 = readAntoine(ff, 1)
                _, lf, jf = ff01
                phs = (-1)**(le + lf)
                
                Jmin, Jmax, parityOK = self._getJRangeForBraKet(ff_sh, ee_sh)
                
                if not parityOK: continue
                
                ## Not necessary the EXCH option, 2 & 3 remain pp-nn & nn-pp
                ta,tb, tc,td = TBME_Reader._JSchemeIndexing[self.t_indx]
                
                if   self.t_indx in (0, 2, 3, 5):
                    tts = (ta,td, tc,tb)
                    assert _ICASE == 0, "Invalid use of the Zeta m.e. (pp-pp)= ICASE 0"
                elif self.t_indx in (1, 4):
                    tts = (ta,tc, td,tb)
                    assert _ICASE == 2, "Invalid use of the Zeta m.e. (pn-pn) -> (pp-nn) ICASE 2, 3"
                else:
                    raise Exception("WTF t index you set?")
                
                for J in range(Jmin, Jmax +1):
                    for TT in (1, -1): ## q1 in F.9, its independent of abcd
                        self.J = J
                        
                        _UVe_fact  = (je + 1) * (jf + 1) * phs
                        _UVe_fact *= self.U_sh[TT][ee] * self.V_sh[TT][ee]
                        _UVe_fact *= self.U_sh[TT][ff] * self.V_sh[TT][ff]
                        
                        print("      * v1: {}{}-ee{} ff{} = [t={}][IC:{}] J={}"
                              .format(ab_sh, cd_sh, ee, ff, self.t_indx, _ICASE, J))
                        
                        bra0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*ee01,  mt= TT),
                                                        QN_1body_jj(*ee01,  mt= TT), self.J)
                        ket0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*ff01,  mt= TT),
                                                        QN_1body_jj(*ff01,  mt= TT), self.J)
                        fix0wf = QN_2body_jj_J_Coupling(QN_1body_jj(*bra01, mt= tts[0]),
                                                        QN_1body_jj(*bra02, mt= tts[1]), _auxJ[0], **k)
                        fix1wf = QN_2body_jj_J_Coupling(QN_1body_jj(*ket01, mt= tts[2]),
                                                        QN_1body_jj(*ket02, mt= tts[3]), _auxJ[0], **k)
                        self.bra   = bra0wf
                        self.ket   = ket0wf
                        self.fix_1 = fix0wf
                        self.fix_2 = fix1wf
                        self.__sh_curr = (ab_sh, cd_sh, ee, ff)
                        
                        norm_ = self.bra.norm() * self.ket.norm() ## Does not appear in the expressions!!
                        v_dd = 0
                        if norm_ != 0:
                            v_dd = self._radial_integral_PiZeta_calculation(2)  
                        v_dd *= _UVe_fact  * factor * norm_
                        
                        ## Save the DD-matrix element with the intermediate value for TESTING
                        self._saveDDMatrixElement (v_dd, ee_sh, ff_sh, fix_1 = ab_sh, fix_2 = cd_sh)
                        self._saveREAMatrixElement(v_dd, ab_sh, cd_sh)
                        
                        if abs(v_dd) > self.NULL_TOLERANCE: 
                            print("      * v1: {}{}-ee{} ff{} = [t={}][IC:{}] {:7.3e}"
                                  .format(ab_sh, cd_sh, ee, ff, self.t_indx, _ICASE, v_dd)) ###
        print()
    
    
    def _convertThePNPN_phMatrixElements(self):
        """
        Convert the pp-nn m.e. already evaluated into the pn-pn.
            For direct V, F, Pi, Zeta and exchanged F and Pi.
        """
        kwargs = {'full_hamiltonian' : False, }
        
        ## Note: The _**_hamil arrays will be overwritten in J-iters_ in Jvals_hamil
        __F_ph_hamil = {3: deepcopy(self._dd_hamil[3]),}
        __P_ph_hamil = deepcopy(self._P_hamil)
        __Z_ph_hamil = deepcopy(self._Z_hamil)
        
        ## clear the hamiltonians pn-pn np-np
        for i, states in enumerate(set(self._ph_sort_pn_dir +
                                       self._ph_sort_pn_exch_F +
                                       self._ph_sort_pn_exch_P)):
            bra, ket = states
            Jmin, Jmax, parOK = self._getJRangeForBraKet(bra, ket)
            
            if bra in self._dd_hamil[3]:
                if ket in self._dd_hamil[3][bra]:
                    for J in range(Jmin, Jmax +1):
                        for t in (1, 4):
                            self._dd_hamil[3][bra][ket][J][t] = 0
            if not almostEqual(self.x0, 1): 
                if bra in self._P_hamil:
                    if ket in self._P_hamil[bra]:
                        for J in range(Jmin, Jmax +1):
                            for t in (1, 4):
                                self._P_hamil[bra][ket][J][t] = 0
                if bra in self._Z_hamil:
                    if ket in self._Z_hamil[bra]:
                        for J in range(Jmin, Jmax +1):
                            for t in (1, 4):
                                self._Z_hamil[bra][ket][J][t] = 0
        
        ## new pn-pn matrix elements from saved __*_hamil
        _IIrange = (1, 2, 3, ) ## 0 (vDD) pn-pn are already calculated
        if almostEqual(self.x0, 1):
            _IIrange = (3, )
        for t in (1, 4):            
            ## direct
            for i, states in enumerate(self._ph_sort_pn_dir):
                ac_sh, db_sh = states
                ad_sh, cb_sh = self._ph_sort_states[i] ## to save
                adcb_sh2 = [(x, readAntoine(x, True)[2]) for x in (*ad_sh, *cb_sh)]
                
                Jmin, Jmax, parOK = self._getJRangeForBraKet(ad_sh, cb_sh)
                if not parOK: continue
                
                for II in _IIrange:
                    for J in range(Jmin, Jmax +1):
                        ## V, F, Pi, Zeta-hamiltonians
                        if II in (0, 3):
                            Jvals_hamil = __F_ph_hamil[II][ac_sh][db_sh] # self._dd_hamil
                        elif II == 1: ## Pi
                            Jvals_hamil = __P_ph_hamil[ac_sh][db_sh]     # self._P_hamil
                        elif II == 2: ## Pi
                            Jvals_hamil = __Z_ph_hamil[ac_sh][db_sh]     # self._Z_hamil
                            
                        args = [*adcb_sh2, J, t, Jvals_hamil]
                        val_pnpn = self._convertPPNNtoPNPN_phme(*args, **kwargs)
                        
                        if II in (0, 3):
                            self._dd_hamil[II][ad_sh][cb_sh][J][t] = val_pnpn
                        elif II == 1: ## Pi
                            self._P_hamil[ad_sh][cb_sh][J][t] = val_pnpn
                        elif II == 2: ## Pi
                            self._Z_hamil[ad_sh][cb_sh][J][t] = val_pnpn
            
            
            ## exchanged F
            for i, states in enumerate(self._ph_sort_pn_exch_F):
                # print(i, states)
                ab_sh, dc_sh = states
                ad_sh, bc_sh = self._ph_sort_exch[i] ## to save
                adbc_sh2 = [(x, readAntoine(x, True)[2]) for x in (*ad_sh, *bc_sh)]
                
                Jmin, Jmax, parOK = self._getJRangeForBraKet(ad_sh, bc_sh)
                if not parOK: continue
                
                for J in range(Jmin, Jmax +1):
                    Jvals_hamil = __F_ph_hamil[3][ab_sh][dc_sh] # self._dd_hamil
                    
                    args = [*adbc_sh2, J, t, Jvals_hamil]
                    val_pnpn = self._convertPPNNtoPNPN_phme(*args, **kwargs)
                    ## TEST: if the matrix element has been evaluated then 
                    #        must have the same value.
                    prev_val = self._dd_hamil[3][ad_sh][bc_sh][J][t]
                    if abs(prev_val) > self.NULL_TOLERANCE:
                        assert abs(prev_val - val_pnpn) < 1.0e-7, \
                            (f"[WARNING] F(pn-pn) matrix element exch already evaluated and do not match "
                             +f"prev[{prev_val:6.5f}] != curr[{val_pnpn:6.5f}] {ad_sh}, {bc_sh}")
                    self._dd_hamil[3][ad_sh][bc_sh][J][t] = val_pnpn
            
            if not almostEqual(self.x0, 1): 
                ## exchanged P
                for i, states in enumerate(self._ph_sort_pn_exch_P):
                    ca_sh, bd_sh = states
                    cb_sh, ad_sh = self._ph_sort_exch_P[i] ## to save
                    cbad_sh2 = [(x, readAntoine(x, True)[2]) for x in (*cb_sh, *ad_sh)]
                    
                    Jmin, Jmax, parOK = self._getJRangeForBraKet(cb_sh, ad_sh)
                    if not parOK: continue
                    
                    for J in range(Jmin, Jmax +1):
                        Jvals_hamil = __P_ph_hamil[ca_sh][bd_sh]  # self._P_hamil
                        
                        args = [*cbad_sh2, J, t, Jvals_hamil]
                        val_pnpn = self._convertPPNNtoPNPN_phme(*args, **kwargs)
                        ## TEST: same if exists previous direct value
                        prev_val = self._P_hamil[cb_sh][ad_sh][J][t]
                        if abs(prev_val) > self.NULL_TOLERANCE:
                            assert abs(prev_val - val_pnpn) < 1.0e-7, \
                                (f"[WARNING] PI(pn-pn) matrix element exch already evaluated and do not match "
                                 +f"prev[{prev_val:6.5f}] != curr[{val_pnpn:6.5f}] {cb_sh}, {ad_sh}")
                        self._P_hamil[cb_sh][ad_sh][J][t] = val_pnpn
            
            
    
    def _radial_integral_calculation(self, IIR):
        """
        Different integrals for the Rearrangement terms
        
            IIR == 0: int(abcd rho_tot(r) r^2dr)
            IIR == 1: int(abcd rho_tot(r)^{-1} rho_t(r) r^2dr)
            IIR == 2: int(abcd rho_tot(r)^{-2} rho_t(r)^2  r^2dr)
            
        """
        ab_sh, cd_sh = self.__sh_curr
        na, la, _ = readAntoine(ab_sh[0], 1)
        nb, lb, _ = readAntoine(ab_sh[1], 1)
        nc, lc, _ = readAntoine(cd_sh[0], 1)
        nd, ld, _ = readAntoine(cd_sh[1], 1)
        
        int_fact = 0.5 * (self.B_LEN**3) / ((2 + self.ALPHA)**1.5)
        radial = (self._radial_2b_wf_memo[(na,la, nb,lb)](self._r) *
                  self._radial_2b_wf_memo[(nc,lc, nd,ld)](self._r) *
                  self._weight_r * int_fact *
                  np.exp((2.+self.ALPHA)* np.power(self._r/self.B_LEN, 2))  )
        angular = np.ones(self._A_DIM) * self._weight_ang
        
        if IIR == 0:
            if self.__sh_curr in self.__radial_integrals_abcd:
                return self.__radial_integrals_abcd.get(self.__sh_curr, 0)
                        
            int_ = np.inner(radial,  np.inner(self._spatial_dens_alp, angular))
            self.__radial_integrals_abcd[self.__sh_curr] = int_
            
        elif IIR == 1:
            ## access to rho_p/n(r), only applies for pppp/nnnn m.e. 
            t = 0 if self.t_indx in (0, 2, 1) else 1
            if self.__sh_curr in self.__radial_int_am1PN_abcd[t]:
                return self.__radial_int_am1PN_abcd[t].get(self.__sh_curr, 0)
                        
            int_ = np.inner(radial,  
                            np.inner(self._spatial_dens_pn[t], angular) *
                            np.inner(self._spatial_dens_alpM1, angular))
            self.__radial_int_am1PN_abcd[t][self.__sh_curr] = int_
        elif IIR == 2:
            ## access to rho_p/n(r), only applies for pppp/nnnn m.e. 
            ## NOTE: The 1 and 4 refer here to he pp-nn matrix elements required
            ##       for the conversion into pn-pn (and nn-pp to np-np).
            if   self.t_indx in (0, 2,   1):
                t = 0
            elif self.t_indx in (5, 3,   4):
                t = 1
            
            if self.__sh_curr in self.__radial_int_am2PN_abcd[t]:
                return self.__radial_int_am2PN_abcd[t].get(self.__sh_curr, 0)
                        
            int_ = np.inner(radial,  
                            np.inner(self._spatial_dens_pn[t], angular)**2 *
                            np.inner(self._spatial_dens_alpM2, angular))
            self.__radial_int_am2PN_abcd[t][self.__sh_curr] = int_
        else:
            raise Exception("Invalid IRR option.", IIR)
        
        return int_
    
    def _FRea0_matrixElement(self, exch=False):
        """
        F-rearrngement matrix elements for rho-term (F.3 F.6)
        """
        J = self.J
        a, b = self.bra.sp_state_1, self.bra.sp_state_2
        if exch:
            c, d = self.ket.sp_state_2, self.ket.sp_state_1
        else:
            c, d = self.ket.sp_state_1, self.ket.sp_state_2
        
        factor  = .5 * self.CONST_t0 * np.sqrt((a.j+1)*(b.j+1)*(c.j+1)*(d.j+1))
        factor *= (-1)**((a.j + b.j + c.j + d.j)//2)
        factor /= (2*self.J + 1) * 4 * np.pi
        
        radial  = self._radial_integral_calculation(0)
        
        if self.t_indx in (0, 5):
            if almostEqual(self.x0, 1): return 0
            
            factor *= 1 - self.x0
            angular = (
                ((-1)**(a.l + b.l + J + (b.j - d.j)/2)) *
                self.__CGC_jab_byJ[J][0][(a.j, b.j)] * 
                self.__CGC_jab_byJ[J][0][(c.j, d.j)] 
                -
                self.__CGC_jab_byJ[J][1][(a.j, b.j)] * 
                self.__CGC_jab_byJ[J][1][(c.j, d.j)] * ((-1)**(d.l+b.l))
            )            
        else:
            angular = (
                ((-1)**((b.j - d.j)/2)) * (1 + self.x0 + (-1)**(a.l + b.l + J)) *
                self.__CGC_jab_byJ[J][0][(a.j, b.j)] * 
                self.__CGC_jab_byJ[J][0][(c.j, d.j)]
                +
                self.x0 * ((-1)**(b.l + d.l)) *
                self.__CGC_jab_byJ[J][1][(a.j, b.j)] * 
                self.__CGC_jab_byJ[J][1][(c.j, d.j)]
            )
            
        val = factor * radial * angular
        return val
    
    def _FRea1_matrixElement(self, exch=False):
        """
        F-rearrngement matrix elements for 1st-derivate rho-term (F.3 F.7)
        """
        J = self.J
        a, b = self.bra.sp_state_1, self.bra.sp_state_2
        if exch:
            c, d = self.ket.sp_state_2, self.ket.sp_state_1
        else:
            c, d = self.ket.sp_state_1, self.ket.sp_state_2
        
        if ((a.l + b.l + J)%2==1) or ((c.l + d.l + J)%2==1): return 0
        
        factor  = 0.5 * self.CONST_t0 * self.ALPHA 
        factor *= np.sqrt((a.j+1)*(b.j+1)*(c.j+1)*(d.j+1))
        factor *= (-1)**((a.j - c.j)/2) 
        factor *= self.__CGC_jab_byJ[J][0][(a.j, b.j)]
        factor *= self.__CGC_jab_byJ[J][0][(c.j, d.j)]
        factor /= (2*self.J + 1) * 4 * np.pi
        
        if self.t_indx in (0, 5):
            radial  = (self._radial_integral_calculation(0) * (self.x0 + 2)
                       -
                       self._radial_integral_calculation(1) * (2*self.x0 + 1))
        else:
            factor *= 3
            radial  = self._radial_integral_calculation(0)  
        
        val = factor * radial
        return val
    
    def _FRea3_matrixElement(self, exch=True):
        """
        F-rearrngement matrix elements for 2nd-derivate rho-term (F.4)
        # Exclude this element for pn-pn type m.e., evaluate only pp-nn to obtain it. 
        """
        J = self.J
        if abs(self.bra.MT) + abs(self.bra.MT) != 2: return 0
        
        a, b = self.bra.sp_state_1, self.bra.sp_state_2
        if exch:
            c, d = self.ket.sp_state_2, self.ket.sp_state_1
        else:
            c, d = self.ket.sp_state_1, self.ket.sp_state_2
        
        if ((a.l + b.l + J)%2==1) or ((c.l + d.l + J)%2==1): return 0
        
        factor  = 0.5 * self.CONST_t0 * self.ALPHA * (self.ALPHA - 1)
        factor *= np.sqrt((a.j+1)*(b.j+1)*(c.j+1)*(d.j+1))
        factor *= (-1)**((a.j - c.j)/2) 
        factor *= self.__CGC_jab_byJ[J][0][(a.j, b.j)]
        factor *= self.__CGC_jab_byJ[J][0][(c.j, d.j)]
        factor /= (2*self.J + 1) * 4 * np.pi
        
        radial  = (
            self._radial_integral_calculation(0) * 0.5 * (1 - self.x0)
            +
            self._radial_integral_calculation(1) * (2*self.x0 + 1)
            -
            self._radial_integral_calculation(2) * (1 - 2*self.x0)
        )
        
        val = factor * radial
        return val
    
    def _radial_integral_PiZeta_calculation(self, IIR):
        """
        Different integrals for the Rearrangement terms, involving the fixed
        states for the sumatory
            
            IIR == 1: int(abcd rho_tot(r)^{-1} e_(r)^2 r^2dr)
            IIR == 2: int(abcd rho_tot(r)^{-2} e_(r)^2 f_(r)^2  r^2dr)
            
        # The integrals are not dependent on the proton or neutron nature of the m.e.
        """
        if   IIR == 1:
            ab_sh, cd_sh, e_sh = self.__sh_curr
            ne, le, _ = readAntoine(e_sh, 1)
        elif IIR == 2:
            ab_sh, cd_sh, e_sh, f_sh = self.__sh_curr
            ne, le, _ = readAntoine(e_sh, 1)
            nf, lf, _ = readAntoine(f_sh, 1)
        else:
            raise Exception(f"Invalid IRR [{IIR}] only 1 and 2 for Pi-Zeta rad. integr")
        
        na, la, _ = readAntoine(ab_sh[0], 1)
        nb, lb, _ = readAntoine(ab_sh[1], 1)
        nc, lc, _ = readAntoine(cd_sh[0], 1)
        nd, ld, _ = readAntoine(cd_sh[1], 1)
        
        int_fact = 0.5 * (self.B_LEN**3) / ((2 + self.ALPHA)**1.5)
        radial = (self._radial_2b_wf_memo[(na,la, nb,lb)](self._r) *
                  self._radial_2b_wf_memo[(nc,lc, nd,ld)](self._r) *
                  self._weight_r * int_fact *
                  np.exp((2.+self.ALPHA)* np.power(self._r/self.B_LEN, 2))  )
        
        radial *= np.power(self._radial_2b_wf_memo[(ne,le, ne,le)](self._r), 2)
        if IIR == 2:
            radial *= np.power(self._radial_2b_wf_memo[(nf,lf, nf,lf)](self._r), 2)
        
        angular = np.ones(self._A_DIM) * self._weight_ang
            
        if   IIR == 1:
            
            if self.__sh_curr in self.__radial_int_am1PN_abcdef:
                return self.__radial_int_am1PN_abcdef.get(self.__sh_curr, 0.0)
                        
            int_ = np.inner(radial, np.inner(self._spatial_dens_alpM1, angular))
            
            self.__radial_int_am1PN_abcdef[self.__sh_curr] = int_
        elif IIR == 2:
            
            if self.__sh_curr in self.__radial_int_am2PN_abcdef:
                return self.__radial_int_am2PN_abcdef.get(self.__sh_curr, 0.0)
                        
            int_ = np.inner(radial, np.inner(self._spatial_dens_alpM2, angular))
            
            self.__radial_int_am2PN_abcdef[self.__sh_curr] = int_
        else:
            raise Exception("Invalid IRR option.", IIR)
        
        return int_
    
    def _saveDDMatrixElement(self, v_dd, bra, ket, fix_1 = None, fix_2 = None):
        """
        Store the different matrix elements, for all the indexes of each dimension
        """
        II = self._ME_TYPE
        J, t = self.J, self.t_indx
        
        if not bra in self._dd_hamil[II]:
            self._dd_hamil[II][bra] = {}
        
        ## complete all the terms for the matrix elements
        if not ket in self._dd_hamil[II][bra]:
            if   II in (0, 3):
                self._dd_hamil[II][bra][ket] = {J : [0.0 for _ in range(6)],}
            ## only J = 0
            elif II == 1:
                dict_ = {}
                for ab_sh in self.sh_states_sorted:
                    dict_[ab_sh] = [0.0 for _ in range(6)]
                self._dd_hamil[II][bra][ket] = dict([(J, deepcopy(dict_)) 
                                                     for J in range(self.j_max+1)])
            elif II == 2:
                dict_ = {}
                for ab_sh in self.sh_states_sorted:
                    dict_[ab_sh] = {}
                    for cd_sh in self.sh_states_sorted:
                        dict_[ab_sh][cd_sh] = [0.0 for _ in range(6)]
                self._dd_hamil[II][bra][ket] = dict([(J, deepcopy(dict_)) 
                                                     for J in range(self.j_max+1)])
        
        if   II in (0, 3):
            if not J in self._dd_hamil[II][bra][ket]:
                self._dd_hamil[II][bra][ket][J] = [0.0 for _ in range(6)]
            if II == 0: 
                self._dd_hamil[II][bra][ket][J][t]  = v_dd
            else: 
                ## F-rea matrix elements have 3 terms += 
                ## (MODIFIED TO avoid repeating m.e. qqnn)
                self._dd_hamil[II][bra][ket][J][t]  = v_dd # +=
        elif II == 1:
            if not fix_1 in self._dd_hamil[II][bra][ket][J]:
                fix_1 = (fix_1[1], fix_1[0])
            self._dd_hamil[II][bra][ket][J][fix_1][t] += v_dd
        elif II == 2:
            if not fix_1 in self._dd_hamil[II][bra][ket][J]:
                fix_1 = (fix_1[1], fix_1[0])
            if not fix_2 in self._dd_hamil[II][bra][ket][J][fix_1]:
                fix_2 = (fix_2[1], fix_2[0])
            self._dd_hamil[II][bra][ket][J][fix_1][fix_2][t] += v_dd
    
    def _saveREAMatrixElement(self, v_dd, bra, ket):
        """
        Store the different matrix elements, for all the indexes of each dimension
        """
        II = self._ME_TYPE
        J, t = self.J, self.t_indx
        
        ATTRIBS_II = ['_Q_ph_hamil', '_P_hamil', '_Z_hamil']
        attr = ATTRIBS_II[II]
        
        if not bra in getattr(self, attr):
            getattr(self, attr)[bra] = {}
        
        ## complete all the terms for the matrix elements
        if not ket in getattr(self, attr)[bra]:
            getattr(self, attr)[bra][ket] = {J : [0.0 for _ in range(6)],}
            ## only J = 0
        
        if not J in getattr(self, attr)[bra][ket]:
            getattr(self, attr)[bra][ket][J] = [0.0 for _ in range(6)]
        if II == 0:
            getattr(self, attr)[bra][ket][J][t] =  v_dd
        else:
            ## MODIFICATION, the Pi/Zeta abcd is implementing the ee, ff series
            getattr(self, attr)[bra][ket][J][t] += v_dd 
        
    
    def _exportHamilsDDTesting(self, tail=''):
        """
        Export the _ME_TYPE internal hamiltonian.
        :tail <optional> appends the string to the final file
        """
        
        filenames_ = ['dd_me_test.2b', 'dd_Pi_test.2b', 'dd_Zeta_test.2b', 
                      'dd_F_me_test.2b', 'dd_FR_me_test.2b']
        II = self._ME_TYPE
        ## testing naming of FR:
        if II == 4 and not all(self._F_REA_2EVAL):
            _s = "{}{}{}".format(*[int(x) for x in self._F_REA_2EVAL])
            filenames_[4] = filenames_[4].replace('.2b', f"_{_s}.2b")
            
        lines = [f'TESTING- type DD [{II}] A,Z={self.A:7.5f},{self.Z:7.5f}   '
                 f'B,ALP,x0,t3[{self.B_LEN:8.5f}, {self.ALPHA:8.5f}, {self.x0:8.5f}, {self.CONST_t0:8.5f}]']
        if II == 1:
            lines.append(" ## Sum over the J values for <ab| fix_st(=cd)|ee>_J on each fixed tuple.")
        if II == 2:
            lines.append(" ## Sum over the J values for <ff| fix_st(=ab, cd)|ee>_J on each fixed tuple.")
        
        for ab, dict_vals_aux in self._dd_hamil[II].items():
            ab = [f'{x:03}' if x==1 else f'{x:}' for x in ab]
            for cd, dict_vals in dict_vals_aux.items():
                cd = [f'{x:03}' if x==1 else f'{x:}' for x in cd]
                Jmin, Jmax = min(dict_vals.keys()),  max(dict_vals.keys())
                
                aux_lines, all_null = [], True
                aux_lines.append(" 0 5 {: >3} {: >3} {: >3} {: >3} {:} {:}"
                                 .format(*ab, *cd, Jmin, Jmax))
                if II in (0, 3, 4):
                    for J, jt_vals in dict_vals.items():
                        all_null *= all([abs(x)<self.NULL_TOLERANCE for x in jt_vals])
                        #aux = self._convertJTmatrixElementsToJScheme(*ab, *cd, J, jt_vals)
                        aux_lines.append("  "+"\t".join([f"{x:15.10f}" for x in jt_vals]))
                    
                else:
                    for a12, vals in dict_vals[0].items():
                        a12 = [f'{x:03}' if x==1 else f'{x:}' for x in a12]
                        if II == 1:
                            all_null *= all([abs(x)<self.NULL_TOLERANCE for x in vals])
                            aux = ' {: >6} {: >6}    '.format(*a12)
                            aux_lines.append(aux+"\t".join([f"{x:15.10f}" for x in vals]))
                        else:
                            for a34, vals2 in vals.items():
                                a34 = [f'{x:03}' if x==1 else f'{x:}' for x in a34]
                                all_null *= all([abs(x)<self.NULL_TOLERANCE for x in vals2])
                                aux = ' {: >6} {: >6} {: >6} {: >6}    '.format(*a12, *a34)
                                aux_lines.append(aux+"\t".join([f"{x:15.10f}" for x in vals2]))
                
                if not all_null: lines = lines + aux_lines
        
        with open(filenames_[II].replace('.2b', tail+'.2b'), 'w+') as f:
            f.write('\n'.join(lines))
    
    def _exportQHamiltonian(self):
        """
            Exporting the hamiltonian
        """
        for II in range(2):
            lines = [f'Q-matrix element DD+REA [ph={not bool(II)}] A,Z={self.A:7.5f},{self.Z:7.5f}   '
                     f'B,ALP,x0,t3[{self.B_LEN:8.5f}, {self.ALPHA:8.5f}, {self.x0:8.5f}, {self.CONST_t0:8.5f}]']
            for ab, dict_vals_aux in self._Q_final_hamil[II].items():
                ab = [f'{x:03}' if x==1 else f'{x:}' for x in ab]
                for cd, dict_vals in dict_vals_aux.items():
                    cd = [f'{x:03}' if x==1 else f'{x:}' for x in cd]
                    Jmin, Jmax = min(dict_vals.keys()),  max(dict_vals.keys())
                    
                    aux_lines, all_null = [], True
                    aux_lines.append(" 0 5 {: >3} {: >3} {: >3} {: >3} {:} {:}"
                                     .format(*ab, *cd, Jmin, Jmax))
                    for J, jt_vals in dict_vals.items():
                        all_null *= all([abs(x)<self.NULL_TOLERANCE for x in jt_vals])
                        #aux = self._convertJTmatrixElementsToJScheme(*ab, *cd, J, jt_vals)
                        aux_lines.append("  "+"\t".join([f"{x:15.10f}" for x in jt_vals]))
                    
                    if not all_null: lines = lines + aux_lines
            
            fn_ = f"Qph_hamilJ_A{A}.2b" if II == 0 else f"Q_hamilJ_A{A}.2b"
            with open(fn_, 'w+') as f:
                f.write('\n'.join(lines))
            
            if II == 1:
                lines2 = [lines[0], '4', ]
                lines2.append('    '+ 
                              ' '.join([  str(len(self.sh_states)), 
                                        *[str(x) for x in self.sh_states]] ))
                lines2.append('    0 0')
                lines2.append('2 {:17.15f}'.format(getattr(self, SHO_Parameters.hbar_omega)))
                with open(fn_.replace('.2b', '.sho'), 'w+') as f:
                    f.write('\n'.join(lines2))
    
    #===========================================================================
    # # COMPUTING ITERATION 
    #===========================================================================
    
    def _isospinExchangeFactor(self):
        """ Matrix element factor associated with the isospin """
        ta,tb = self.bra.sp_state_1.m_t, self.bra.sp_state_2.m_t
        tc,td = self.ket.sp_state_1.m_t, self.ket.sp_state_2.m_t
        
        X_ac_bd  = (1       + self.x0M)*((ta==tc)*(tb==td))
        X_ac_bd -= (self.x0 + self.x0H)*((ta==td)*(tb==tc))
        X_ad_bc  = (1       + self.x0M)*((ta==td)*(tb==tc)) 
        X_ad_bc -= (self.x0 + self.x0H)*((ta==tc)*(tb==td))
        
        self.X_ac_bd = X_ac_bd
        self.X_ad_bc = X_ad_bc
        
    def _isospinExchangeFactor_PH(self, exchange=False):
        """ Matrix element factor associated with the isospin """
        if exchange:
            ta,tb = self.bra.sp_state_1.m_t, self.ket.sp_state_1.m_t
            tc,td = self.bra.sp_state_2.m_t, self.ket.sp_state_2.m_t
        else:
            ta,tb = self.bra.sp_state_1.m_t, self.ket.sp_state_2.m_t
            tc,td = self.bra.sp_state_2.m_t, self.ket.sp_state_1.m_t
        
        X_ac_bd  = (1       + self.x0M)*((ta==tc)*(tb==td))
        X_ac_bd -= (self.x0 + self.x0H)*((ta==td)*(tb==tc))
        X_ad_bc  = (1       + self.x0M)*((ta==td)*(tb==tc)) 
        X_ad_bc -= (self.x0 + self.x0H)*((ta==tc)*(tb==td))
        
        self.X_ac_bd = X_ac_bd
        self.X_ad_bc = X_ad_bc
    
    def _radialAngularDDIntegral(self, a_sp, b_sp, c_sp, d_sp): ## mja, mjb, mjc, mjd
        """
        """        
        # if almostEqual(self.PARAMS_FORCE[dd_p.x0], 1, 1.e-4):
        #     ## this condition omits the pppp / nnnn in the x0=1 case.
        if abs(self.X_ac_bd) + abs(self.X_ad_bc) < self.NULL_TOLERANCE: 
            return 0.0
        
        _, la, ja, mja = a_sp
        _, lb, jb, mjb = b_sp
        _, lc, jc, mjc = c_sp
        _, ld, jd, mjd = d_sp
        
        indx_a = angular_Y_KM_index(ja, mja, True)        
        indx_b = angular_Y_KM_index(jb, mjb, True)
        indx_c = angular_Y_KM_index(jc, mjc, True)        
        indx_d = angular_Y_KM_index(jd, mjd, True)
        
        aux_d   = np.zeros(self._A_DIM)
        aux_e   = np.zeros(self._A_DIM)
        ## DIR
        if not self.isNullValue(self.X_ac_bd):
            for K1 in range(abs(ja-jc)//2, (ja+jc)//2 +1):
                M1 = (mjc - mja) // 2
                if ((abs(M1) > K1) or ((K1 + la + lc) % 2 == 1)): continue
                
                indx_K1  = angular_Y_KM_index(K1, M1, False)
                key_acK = _angular_Y_KM_memo_accessor(indx_a,indx_c,indx_K1)
                
                for K2 in range(abs(jb-jd)//2, (jb+jd)//2 +1):
                    M2 = (mjd - mjb) // 2
                    if ((abs(M2) > K2) or ((K2 + lb + ld) % 2 == 1)): continue
                
                    indx_K2 = angular_Y_KM_index(K2, M2, False)
                    key_bdK = _angular_Y_KM_memo_accessor(indx_b,indx_d,indx_K2)
                    
                    c_ang    = _angular_Y_KM_me_memo.get(key_acK, 0)
                    c_ang   *= _angular_Y_KM_me_memo.get(key_bdK, 0)
                    sph_har  = copy(self._sph_harmonic_memo[indx_K1])
                    sph_har *= self._sph_harmonic_memo[indx_K2]
                    aux_d    = aux_d + (c_ang * sph_har)
        
        ## EXCH
        if not self.isNullValue(self.X_ad_bc):
            for K1 in range(abs(ja-jd)//2, (ja+jd)//2 +1):
                M1 = (mjd - mja) // 2
                if ((abs(M1) > K1) or ((K1 + la + ld) % 2 == 1)): continue
                
                indx_K1  = angular_Y_KM_index(K1, M1, False)
                key_adK = _angular_Y_KM_memo_accessor(indx_a,indx_d,indx_K1)
                
                for K2 in range(abs(jb-jc)//2, (jb+jc)//2 +1):
                    M2 = (mjc - mjb) // 2
                    if ((abs(M2) > K2) or ((K2 + lb + lc) % 2 == 1)): continue
                
                    indx_K2 = angular_Y_KM_index(K2, M2, False)
                    key_bcK = _angular_Y_KM_memo_accessor(indx_b,indx_c,indx_K2)
                
                    c_ang    = _angular_Y_KM_me_memo.get(key_adK, 0)
                    c_ang   *= _angular_Y_KM_me_memo.get(key_bcK, 0)
                    sph_har  = copy(self._sph_harmonic_memo[indx_K1])
                    sph_har *= self._sph_harmonic_memo[indx_K2]
                    aux_e    = aux_e + (c_ang * sph_har)
        
        angular = (self.X_ac_bd * aux_d) - (self.X_ad_bc * aux_e)
        angular = angular * self._weight_ang
        
        ## Same loop but using a the matrix product  [rad]*[dens_pow]*[ang]
        
        if   self._ME_TYPE in (0, 3):
            me_val = np.inner(self._curr_radial, 
                              np.inner(self._spatial_dens_alp, angular))
        elif self._ME_TYPE == 1:
            me_val = np.inner(self._curr_radial, 
                              np.inner(self._spatial_dens_alpM1, angular))
        elif self._ME_TYPE == 2:
            me_val = np.inner(self._curr_radial, 
                              np.inner(self._spatial_dens_alpM2, angular))
        
        me_val *= (4 * np.pi) if self.USING_LEBEDEV else np.pi
        return me_val
    
    def _antisymmetrized_J_element(self):
        """
        DD integral for the element uncoupled in LS-T scheme.
        """
        self._isospinExchangeFactor()
        
        integral_factor  = self.CONST_t0 * 0.5 * (self.B_LEN**3)
        integral_factor /= (2 + self.ALPHA)**1.5
        
        if   self._ME_TYPE >= 1: integral_factor *= self.ALPHA
        elif self._ME_TYPE == 2: integral_factor *= (self.ALPHA - 1)
        
        na, la, ja = self.bra.n1, self.bra.l1, self.bra.j1
        nb, lb, jb = self.bra.n2, self.bra.l2, self.bra.j2
        nc, lc, jc = self.ket.n1, self.ket.l1, self.ket.j1
        nd, ld, jd = self.ket.n2, self.ket.l2, self.ket.j2
                
        self._curr_radial = (
            self._radial_2b_wf_memo[(na,la, nb,lb)](self._r) *
            self._radial_2b_wf_memo[(nc,lc, nd,ld)](self._r) *
            self._weight_r * 
            np.exp((2.+self.ALPHA)* np.power(self._r/self.B_LEN, 2))  )
        
        if   self._ME_TYPE == 0:
            pass
        elif self._ME_TYPE == 1:
            n1, l1, _ = readAntoine(self.fix_1[0], True)
            n2, l2, _ = readAntoine(self.fix_1[1], True)
            self._curr_radial *= self._radial_2b_wf_memo[(n1,l1, n2,l2)](self._r)
        elif self._ME_TYPE == 2:
            n1, l1, _ = readAntoine(self.fix_1[0], True)
            n2, l2, _ = readAntoine(self.fix_1[1], True)
            n3, l3, _ = readAntoine(self.fix_2[0], True)
            n4, l4, _ = readAntoine(self.fix_2[1], True)
            self._curr_radial *= (
                self._radial_2b_wf_memo[(n1,l1, n2,l2)](self._r) * 
                self._radial_2b_wf_memo[(n3,l3, n4,l4)](self._r) )
        
        ma_valid = [m for m in range(-self.bra.j1, self.bra.j1 +1, 2)]
        mb_valid = [m for m in range(-self.bra.j2, self.bra.j2 +1, 2)]
        
        mc_valid = [m for m in range(-self.ket.j1, self.ket.j1 +1, 2)]
        md_valid = [m for m in range(-self.ket.j2, self.ket.j2 +1, 2)]
        
        list_me = {}
        me_value = 0.0
        for ma in ma_valid:
            for mb in mb_valid:
                if self.bra.M != (ma+mb)//2: continue
                
                args_b = (ja/2, jb/2, self.bra.J,
                          ma/2, mb/2, self.bra.M)
                ccg_b = safe_clebsch_gordan(*args_b)
                if self.isNullValue(ccg_b): continue
                
                for mc in mc_valid:
                    for md in md_valid:
                        
                        if self.ket.M != (mc+md)//2: continue
                        args_k = (jc/2, jd/2, self.ket.J,
                                  mc/2, md/2, self.ket.M)
                        ccg_k = safe_clebsch_gordan(*args_k)
                        ang_recoup = ccg_b * ccg_k
                        
                        if self.isNullValue(ang_recoup): continue
                        
                        a, b = (na, la, ja, ma), (nb, lb, jb, mb)
                        c, d = (nc, lc, jc, mc), (nd, ld, jd, md)
                        
                        dd_integral = self._radialAngularDDIntegral(a, b, c, d)
                        me_value += (ang_recoup * dd_integral)
                        list_me[(ma, mb, mc, md)] = dd_integral, ang_recoup
        
        if (not self.t_indx in (0, 5)) and abs(np.real(me_value)) > 1.0e-22:
            if ((na,la,ja, nb,lb,jb, nc,lc,jc, nd,ld,jd) ==
                ( 1, 0, 1,  1, 0, 1,  0, 0, 1,  0, 0, 1)):
                _ = 0
        me_value *= integral_factor
        if abs(np.imag(me_value)) > self.NULL_TOLERANCE:
            raise MatrixElementException(f"Imaginary vDD matrix element [{self._ME_TYPE}]")
        return np.real(me_value)
    
    def _antisymmetrizedPH_J_element(self, exchange=False):
        """
        Evaluating the PH-uncoupled DD matrix element, running on M(bra)==M(ket)
        """
        self._isospinExchangeFactor_PH(exchange)    
        integral_factor  = self.CONST_t0 * 0.5 * (self.B_LEN**3)
        integral_factor /= (2 + self.ALPHA)**1.5
        
        na, la, ja = self.bra.n1, self.bra.l1, self.bra.j1
        nb, lb, jb = self.bra.n2, self.bra.l2, self.bra.j2
        if exchange:
            n2, l2, j2 = self.ket.n1, self.ket.l1, self.ket.j1
            n1, l1, j1 = self.ket.n2, self.ket.l2, self.ket.j2
        else:
            n1, l1, j1 = self.ket.n1, self.ket.l1, self.ket.j1
            n2, l2, j2 = self.ket.n2, self.ket.l2, self.ket.j2
                
        self._curr_radial = (
            self._radial_2b_wf_memo[(na,la, n1,l1)](self._r) *
            self._radial_2b_wf_memo[(nb,lb, n2,l2)](self._r) *
            self._weight_r * 
            np.exp((2.+self.ALPHA)* np.power(self._r/self.B_LEN, 2))  )
        
        ma_valid = [m for m in range(-self.bra.j1, self.bra.j1 +1, 2)]
        mb_valid = [m for m in range(-self.bra.j2, self.bra.j2 +1, 2)]
        
        m1_valid = [m for m in range(-j1, j1 +1, 2)]
        m2_valid = [m for m in range(-j2, j2 +1, 2)]
        
        me_value = 0.0
        for ma in ma_valid:
            for mb in mb_valid:
                if self.bra.M != (ma+mb)//2: continue
                
                args_b = (ja/2, jb/2, self.bra.J,
                          ma/2, mb/2, self.bra.M)
                ccg_b = safe_clebsch_gordan(*args_b)
                if self.isNullValue(ccg_b): continue
                
                for m1 in m1_valid:
                    for m2 in m2_valid:
                        
                        if self.ket.M != (m1+m2)//2: continue
                        args_k = (j1/2, j1/2, self.ket.J,
                                  m1/2, m2/2, self.ket.M)
                        ccg_k = safe_clebsch_gordan(*args_k)
                        ang_recoup = ccg_b * ccg_k
                        
                        if self.isNullValue(ang_recoup): continue
                        
                        a, d2 = (na, la, ja, +ma), (n2, l2, j2, -m2)
                        b2, c = (nb, lb, jb, -mb), (n1, l1, j1, +m1)
                        
                        dd_integral = self._radialAngularDDIntegral(a, d2, b2, c)
                        me_value += (ang_recoup * dd_integral)
                
        me_value *= integral_factor
        if abs(np.imag(me_value)) > self.NULL_TOLERANCE:
            raise MatrixElementException(f"Imaginary vDD matrix element [{self._ME_TYPE}]")
        return np.real(me_value)
    
    def _ZetaRearrange_J_element(self, bra, ket):
        """
            Zeta(abcd, J) element, computing the U-V matrices
        """
        a_sh, b_sh, c_sh, d_sh = *bra, *ket
        
        a = self.sh_states_obj[a_sh]
        b = self.sh_states_obj[b_sh]
        c = self.sh_states_obj[c_sh]
        d = self.sh_states_obj[d_sh]
        
        if ((a.l + b.l + self.J)%2 == 1): return 0
        if ((c.l + d.l + self.J)%2 == 1): return 0
        
        factor  = safe_clebsch_gordan(a.j/2, b.j/2, self.J,  1/2, -1/2, 0)
        factor *= safe_clebsch_gordan(c.j/2, c.j/2, self.J,  1/2, -1/2, 0)
        if self.isNullValue(factor): return 0
        
        factor *= (
            np.sqrt((a.j+1)*(b.j+1)*(c.j+1)*(d.j+1)) * ((-1)**((a.j-c.j)//2)) 
            / 
            (64*(np.pi**2)*(2*self.J + 1)))
        
        val = 0
        for e_sh, e in self.sh_states_obj.items():
            for f_sh, f in self.sh_states_obj.items():
                
                #ta,tb,tc,td = TBME_Reader._JSchemeIndexing[self.t_indx]
                
                uv = 0.0
                for te in (1, -1):
                    for tf in (1, -1):
                        ## T-ef do not follow the outside t indexing, since
                        ## matrix elements are T=0
                        #TODO: VERIFY this!
                        uv += (self.U_sh[te][e_sh] * self.V_sh[te][e_sh] *
                               self.U_sh[tf][f_sh] * self.V_sh[tf][f_sh] )
                        
                me = self.__get_ddME(2, *bra, *ket, self.J, self.t_indx,
                                     ee=(e_sh, e_sh), ff=(f_sh, f_sh) )
                
                me *= np.sqrt((e.j + 1)*(f.j + 1)) * uv
        
        val *= factor
        return val
    
    def _PiRearrange_J_element(self, bra, ket):
        """
            Pi(abcd, J) element, computing the U-V matrices
        """
        a_sh, b_sh, c_sh, d_sh = *bra, *ket
        
        a = self.sh_states_obj[a_sh]
        b = self.sh_states_obj[b_sh]
        c = self.sh_states_obj[c_sh]
        d = self.sh_states_obj[d_sh]
        
        if (self.J != 0): return 0
        if ((a.j != b.j) or (c.j != d.j) or (c.l != d.l)): return 0
        
        factor  = np.sqrt(c.j + 1) / (8*np.pi)
        
        val = 0
        for e_sh, e in self.sh_states_obj.items():
                
            #ta,tb,tc,td = TBME_Reader._JSchemeIndexing[self.t_indx]
            
            uv = 0.0
            for te in (1, -1):
                ## T-ef do not follow the outside t indexing, since
                ## matrix elements are T=0
                #TODO: VERIFY this!
                uv += self.U_sh[te][e_sh] * self.V_sh[te][e_sh]
                    
            me = self.__get_ddME(1, *bra, *ket, self.J, self.t_indx, 
                                 ee=(e_sh, e_sh))
            
            me *= np.sqrt((e.j + 1)) * uv
        
        val *= factor
        return val
    
    def _calculateBulkMatrixElements(self):
        """
        Compute the U-V dependent matrix elements, called Pi and Zeta in the
        article (3.14, .15)
        """
        for ia, a_sh in enumerate(self.sh_states):
            for _, b_sh in enumerate(self.sh_states[ia:]):
                
                Jmin0, Jmax0 = getJrangeFor2ShellStates(a_sh, b_sh)
                
                for ic, c_sh in enumerate(self.sh_states):
                    for _, d_sh in enumerate(self.sh_states[ic:]):
                        bra, ket = (a_sh, b_sh), (c_sh, d_sh)
                        
                        parL = [self.sh_states_obj[x].l for x in (a_sh, b_sh, c_sh, d_sh)]
                        if sum(parL)%2 != 0: continue
                        Jmin, Jmax = getJrangeFor2ShellStates(c_sh, d_sh)
                        if Jmin > Jmax0 or Jmax < Jmin0: continue
                        Jmin, Jmax = max(Jmin, Jmin0), min(Jmax, Jmax0)
                        
                        self._ME_TYPE = 2
                        ## Zeta functions:
                        for J in range(Jmin, Jmax +1):
                            self.J = J
                            for t in range(6):
                                self.t_indx = t
                                
                                v_val = self._ZetaRearrange_J_element(bra, ket)
                                
                                self._saveREAMatrixElement(v_val, bra, ket)
                        
                        ## Pi values (only J=0)
                        if Jmin == 0:
                            self.J = J
                            self._ME_TYPE = 1
                            for t in range(6):
                                self.t_indx = t
                                
                                v_val = self._PiRearrange_J_element(bra, ket)
                                self._saveREAMatrixElement(v_val, bra, ket)
                                
                                if bra == ket: continue
                                
                                v_val = self._PiRearrange_J_element(ket, bra)
                                self._saveREAMatrixElement(v_val, bra, ket)
                        
        
        self._PiZeta_hamil[0] = deepcopy(self._P_hamil)
        self._PiZeta_hamil[1] = deepcopy(self._Z_hamil)
    
    def __get_ddME(self, II, a, b, c, d, J, t, ee=None, ff=None):
        """
        Access safely to the DD and rearrangement elements
        : ee / ff are (e, e) sh states
        """
        assert II in (0, 1, 2, 3, ), "?? invalid index [0:3]"
        val = 0
        try:
            if (a, b) in self._dd_hamil[II]:
                if (c, d) in self._dd_hamil[II][(a, b)]:
                    # only J=0 for these elements, and all t values saved
                    if II == 1:
                        if not (ee in self._dd_hamil[II][(a, b)][(c, d)][0]): 
                            return val
                        val = self._dd_hamil[II][(a, b)][(c, d)][0][ee][t] 
                    if II == 2:
                        if not (ee in self._dd_hamil[II][(a, b)][(c, d)][0]): 
                            return val
                        if (ff in self._dd_hamil[II][(a, b)][(c, d)][0][ee]):
                            val = self._dd_hamil[II][(a, b)][(c, d)][0][ee][ff][t]                        
                    else: 
                        if J in self._dd_hamil[II][(a, b)][(c, d)]:
                            val = self._dd_hamil[II][(a, b)][(c, d)][J][t]
        except KeyError as ke:
            print(f"Error: _dd_hamil[II:{II}][(a, b)][(c, d)][ee][ff][J:{J}][t:{t}]=", 
                  a, b, c, d, ee, ff)
            raise ke
        return val
    
    def __get_ReaME(self, I2, a, b, c, d, J, t):
        """
        Same access but only for final PI(0) Zeta(1) Rearrange matrix elements
        """
        assert I2 in (0, 1)
        xx = 0
        if (a, b) in self._PiZeta_hamil[I2]:
            if (c, d) in self._PiZeta_hamil[I2][(a, b)]:
                if J  in self._PiZeta_hamil[I2][(a, b)][(c, d)]:
                    xx = self._PiZeta_hamil[I2][(a, b)][(c, d)][J][t]
        return xx
    
    def _computeRearrangementMatrixElements(self):
        """
        Performing the last U-V operations on the m.e <ab|v|cd(J)>, F, Pi, Eta
        Computes Q(abcd | J)
        """
        print("\n  [FINAL] COMPUTING REARRANGEMENT - Q MATRIX-ELEMENTS \n")
        self._ME_TYPE = 0
        self.printUVs()
        if self.APPLY_PH_OCCUPATION_DD_2BME: _dd_damped = deepcopy(self._dd_hamil[0])
        
        for i, states in enumerate(self._ph_sort_states):
            ad_sh, cb_sh = states
            a, d = ad_sh
            c, b = cb_sh
            
            b_sh = self.sh_states_obj[b]
            c_sh = self.sh_states_obj[c]
                        
            Jmin, Jmax, parityOK = self._getJRangeForBraKet(ad_sh, cb_sh)
            if not parityOK: continue
            
            for J in range(Jmin, Jmax +1):
                self.J = J
                for t in range(6):
                    self.t_indx = t
                    ta,td,tc,tb = TBME_Reader._JSchemeIndexing[t]
                    
                    #### V_DD part
                    aux0 = 0
                    if self.APPLY_PH_OCCUPATION_DD_2BME:
                        aux0 = ((self.U_sh[ta][a] * self.U_sh[td][d] *
                                 self.U_sh[tc][c] * self.U_sh[tb][b])
                                 +
                                (self.V_sh[ta][a] * self.V_sh[td][d] *
                                 self.V_sh[tc][c] * self.V_sh[td][b]))
                        aux0 *= self.__get_ddME(0, a, d, c, b, J, t)
                        _dd_damped[(a, d)][(c, b)][J][t] = aux0
                    
                    # if (ad_sh, cb_sh) ==  ((1, 203), (1, 205)) and t in (1, 4): 
                    #     _ = 0
                    
                    #### F_DD ph part                                
                    aux1 = ((self.U_sh[ta][a] * self.V_sh[td][d] *
                             self.U_sh[tc][c] * self.V_sh[tb][b])
                             +
                            (self.V_sh[ta][a] * self.U_sh[td][d] *
                             self.V_sh[tc][c] * self.U_sh[tb][b]))
                    aux1 *= self.__get_ddME(3, a, d, c, b, J, t)
                           
                    aux  = ((self.U_sh[ta][a] * self.V_sh[td][d] *
                             self.V_sh[tc][c] * self.U_sh[tb][b])
                             +
                            (self.V_sh[ta][a] * self.U_sh[td][d] *
                             self.U_sh[tc][c] * self.V_sh[tb][b]))
                    aux  *= self.__get_ddME(3, a, d, b, c, J, t)
                    
                    aux1 += ((-1)**((c_sh.j - b_sh.j)/2 + J)) * aux
                    
                    #### 2nd Derivative V_DD part 
                    phs = (-1)**J
                    aux2  = ((            self.U_sh[ta][a] * self.V_sh[td][d]
                                 + (phs * self.V_sh[ta][a] * self.U_sh[td][d]))
                             * (          self.U_sh[tc][c] * self.V_sh[tb][b]
                                 + (phs * self.V_sh[tc][c] * self.U_sh[tb][b])) )
                             
                    aux2 *= self.__get_ReaME(1, a, d, c, b, J, t)
                    
                    #### 1st Derivative V_DD part
                    aux3 = 0
                    if J == 0:
                        aux30 = (
                            (self.U_sh[ta][a] * self.U_sh[td][d] -
                             self.V_sh[ta][a] * self.V_sh[td][d])
                            *
                            (self.U_sh[tc][c] * self.V_sh[tb][b] +
                             self.V_sh[tc][c] * self.U_sh[tb][b]))
                        aux3 += aux30 * self.__get_ReaME(0, a, d, c, b, J, t)
                        
                        aux31 = (
                            (self.U_sh[tc][c] * self.U_sh[tb][b] -
                             self.V_sh[tc][c] * self.V_sh[tb][b])
                            *
                            (self.U_sh[ta][a] * self.V_sh[td][d] +
                             self.V_sh[ta][a] * self.U_sh[td][d]))
                        aux3 += aux31 * self.__get_ReaME(0, c, b, a, d, J, t)
                    
                    Q_dd = aux0 + aux1 + aux2 + aux3
                    
                    self._saveREAMatrixElement(Q_dd, (a,d), (c,b))
        
        if self.APPLY_PH_OCCUPATION_DD_2BME:
            self._dd_hamil[0] = deepcopy(_dd_damped)
            self._ME_TYPE = 0
            self._exportHamilsDDTesting(tail='_ph_uv')
    
    def _convertFinalPHMatrixElements(self):
        """
        create Q_final_hamil for the 2b-matrix elements
        """
        aux_hamil = {}
        kwargs = {'full_hamiltonian': False, }
        list_ = [0 for _ in range(6)]
        
        for i, states in enumerate(self._ph_sort_states):
            ad_sh, cb_sh = states
            bra2save, ket2save = self._2b_sort_states[i]
                
            bra01, bra02 = readAntoine(ad_sh[0], 1), readAntoine(ad_sh[1], 1)
            ket01, ket02 = readAntoine(cb_sh[0], 1), readAntoine(cb_sh[1], 1)
            
            if not bra2save in aux_hamil: 
                aux_hamil[bra2save] = {}
            
            Jmin, Jmax, parityOK = self._getJRangeForBraKet(bra2save , ket2save)
            if not parityOK: continue
            
            if not ket2save in aux_hamil[bra2save]:    
                j_dict = dict([(J, deepcopy(list_)) for J in range(Jmin, Jmax + 1)])
                aux_hamil[bra2save][ket2save] = deepcopy(j_dict)
            
            # Element DD
            for J in range(Jmin, Jmax +1):
                self.J = J
                for t in range(6):
                    ta,tb,tc,td = TBME_Reader._JSchemeIndexing[t]
                    bra2wf = QN_2body_jj_J_Coupling(QN_1body_jj(*bra01, mt= ta),
                                                    QN_1body_jj(*ket02, mt= tb), J)
                    ket2wf = QN_2body_jj_J_Coupling(QN_1body_jj(*ket01, mt= tc),
                                                    QN_1body_jj(*bra02, mt= td), J)
                    if almostEqual(bra2wf.norm() * ket2wf.norm(),  0.0):
                        continue
                    
                    args = [
                        (bra2save[0], bra01[2]),  (bra2save[1], ket02[2]),
                        (ket2save[0], ket01[2]),  (ket2save[1], bra02[2]), 
                        J, t, 
                        self._Q_ph_hamil[ad_sh][cb_sh]
                    ]
                    
                    val = self.pandya_j_scheme_transformation(*args, **kwargs)
                    aux_hamil[bra2save][ket2save][J][t] = val
    
        self._Q_final_hamil[1] = deepcopy(aux_hamil)
    
    def _appendDD2bMEToQMatrixElements(self):
        """ 
        Append the DD matrix elements (no dependence on U-V)
        """
        for states in self._2b_sort_states:
            bra, ket = states
            if not bra in self._dd_hamil[0]: continue
            if not ket in self._dd_hamil[0][bra]: continue
            
            if not bra in self._Q_final_hamil[1]:
                self._Q_final_hamil[1][bra] = {ket: dict(),}
            if not ket in self._Q_final_hamil[1][bra]:
                self._Q_final_hamil[1][bra][ket] = dict()
                
            for J, t_vals in self._dd_hamil[0][bra][ket].items():
                if not J in self._Q_final_hamil[1][bra][ket]:
                    self._Q_final_hamil[1][bra][ket][J] = deepcopy(t_vals)
                    continue
                
                for t, val in enumerate(t_vals):
                    self._Q_final_hamil[1][bra][ket][J][t] += val 
                
        
#===============================================================================
##  MAIN
#===============================================================================
if __name__ == '__main__':
    
    wf_filename = 'final_wf_16O_N3.txt' #'final_wf_30Mg.txt' #
    A = 24 # 16 # 
    
    # for A in range(16, 40+1, 4):
    wf_filename = f'results_wf/final_wf_A{A}.txt'
    hbaromega = 45*A**(-1/3) - 25*A**(-2/3)
    b_len     = Constants.HBAR_C / np.sqrt(Constants.M_MEAN * hbaromega)
    
    kwargs = {
        _DD_pe.constant : {'value': 1390.6,},
        _DD_pe.x0       : {'value':    1,},
        _DD_pe.alpha    : {'value':    1/3,},
        _DD_pe.file     : {'name' : wf_filename,},
        # x0H   = {'value':      1,},            #     Heisenberg -P^t     
        # x0M   = {'value':      1,},            #     Majorana   -P^t*P^s
        # core  = <core protons='12' neutrons ='20' core_b_len='1.89'/>
        # integration = 'integration'
        SHO_Parameters.b_length: b_len,
        SHO_Parameters.hbar_omega: hbaromega,
    }
    
    _runner = RearrangementTBMESpherical_Runner()
    
    #===========================================================================
    ## NOTE: This option includes the v_DD term weighted from u,v_ (eq. )
    ##       The expression in the article makes no sense for the occupied-empty
    ##       matrix elements, so it only considers it for fully occupied/empty 
    ##       states. 
    ##       OCCUPIED_TOL set the criterion on the U to consider a state empty.
    #===========================================================================
    # _runner.OCCUPIED_TOL = 0.90
    # _runner.APPLY_PH_OCCUPATION_DD_2BME = True
    
    
    _runner.setInteractionParameters(**kwargs)
    _runner.run()
    _runner.printUVs()
    hamil = _runner.getMatrixElements()
    
    print("[DONE] Script- for Rearrangement Matrix elements")
    
    