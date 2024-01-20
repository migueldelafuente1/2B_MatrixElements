'''
Created on 29 dic 2023

@author: delafuente
'''

import numpy as np
from sympy import S

from helpers.Helpers import safe_3j_symbols, almostEqual, safe_clebsch_gordan,\
    readAntoine

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    MatrixElementException, _TwoBodyMatrixElement_Antisym_JCoupled
from helpers.Enums import CouplingSchemeEnum, AttributeArgs,\
    SHO_Parameters, DensityDependentParameters
from helpers.Log import XLog
from helpers.integrals import _RadialDensityDependentFermi
from helpers.WaveFunctions import QN_1body_radial, \
    QN_2body_jj_J_Coupling, QN_1body_jj
from helpers.mathFunctionsHelper import _buildAngularYCoeffsArray,\
    _buildRadialFunctionsArray, angular_Y_KM_index, sphericalHarmonic,\
    _angular_Y_KM_memo_accessor, _angular_Y_KM_me_memo
from helpers import SCIPY_INSTALLED
from copy import deepcopy, copy

from helpers.Enums import DensityDependentParameters as dd_p
from helpers.Enums import AttributeArgs as atrE

# class DensityDependentForce_JTScheme(_TwoBodyMatrixElement_Antisym_JTCoupled):
class DensityDependentForce_JTScheme(_TwoBodyMatrixElement_JTCoupled):

    """
    Density term based on Fermi density distribution, (ordered filled up to A 
    mass number). 
    """
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    
    _BREAK_ISOSPIN = False
    
    _R_DIM = 12
    _A_DIM = None
    _OMEGA = None
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """
        Arguments for a radial potential form V(r; mu_length, constant, n, ...)
        
        :b_length 
        :hbar_omega
        :potential       <str> in PotentialForms Enumeration
        :mu_length       <float>  fm
        :constant        <float>  MeV
        :n_power         <int>
        
        method bypasses calling from main or io_manager
        """
        # Refresh the Force parameters
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        _b = SHO_Parameters.b_length
        _a = SHO_Parameters.A_Mass
        _z = SHO_Parameters.Z
        
        cls.PARAMS_SHO[_b] = float(kwargs.get(_b))
        cls.PARAMS_SHO[_a] = int(kwargs.get(_a))
        z = kwargs.get(_z)
        cls.PARAMS_SHO[_z] = int(z) if z else None
        
        cls.PARAMS_FORCE = {}
        cls.PARAMS_CORE  = {}
        cls._has_core_b_lenght = False
        fa = atrE.ForceArgs
        ## To avoid exception in DD sho aproximation
        if not dd_p.file in kwargs:
            kwargs[dd_p.file] = None
        
        for param in dd_p.members():
            aux = kwargs.get(param, None)
            if param == dd_p.core:
                if aux==None: aux = {}
                b = aux.get(fa.DensDep.core_b_len, None)
                if (b != None) and (b != ''):
                    cls.PARAMS_CORE[fa.DensDep.core_b_len] = float(b)
                    cls._has_core_b_lenght = True
                _z = aux.get(fa.DensDep.protons, '0')
                _n = aux.get(fa.DensDep.neutrons,'0')
                if '.' in _z or '.' in _n: 
                    print("[Error] Set up d.d force core Z or N number is float["
                          ,_z,_n,"] use only integers")
                    if '.' in _z or '.' in _z: 
                        raise MatrixElementException("Invalid N/Z core. Stop")
                
                _z, _n = abs(int(_z)), abs(int(_n)) 
                # in case somebody set them <0
                if _z == 0 and _n == 0:
                    print("[Warning] Cannot set A=0 core, Z=N=0, so PARAMS_CORE",
                          "ignores arguments given, A,Z come form SHO_params:",
                          cls.PARAMS_SHO)
                    continue
                cls.PARAMS_CORE[fa.DensDep.protons ] = int(_z)
                cls.PARAMS_CORE[fa.DensDep.neutrons] = int(_n)  
            
            elif param in (dd_p.file, dd_p.integration):
                # Entries for DFromFile but not for BaseDensity, 
                # "file" is mandatory, "integration" is optional (both dict like)
                if param == dd_p.file:
                    if (cls.__name__==DensityDependentForceFromFile_JScheme.__name__
                        and (aux==None)):
                        raise MatrixElementException(
                            f"Required tag parameter f[{param}] for file importing.")
                    if isinstance(aux, dict):
                        if AttributeArgs.name in aux:
                            aux = aux[AttributeArgs.name]
                    cls.PARAMS_FORCE[param] = aux ## is string
                else:
                    if aux == None: continue
                    new_rdim = aux.get(fa.DensDep.r_dim, None)
                    if new_rdim : cls._R_DIM = int(new_rdim)
                    new_omeg = aux.get(fa.DensDep.omega_ord, None)
                    if new_omeg : cls._OMEGA = int(new_omeg)
                    # _OMEGA is not used in the SHO density matrix element.
            else:
                assert aux != None, f"Required tag parameter [{param}], not given"
                if isinstance(aux, dict):
                    aux = float(aux[AttributeArgs.value])
                cls.PARAMS_FORCE[param] = aux
                
        #cls.PARAMS_FORCE[CentralMEParameters.potential] = PotentialForms.Gaussian
    
    def _validKetTotalAngularMomentums(self):
        return (self.L_bra, )
    
    def _validKetTotalSpins(self):
        return (self.S_bra, )
    
    def _LScoupled_MatrixElement(self):
        
        phs = ((-1)**self.S_bra)
        fact = 1 - (phs * self.PARAMS_FORCE[DensityDependentParameters.x0])
        
        ## Antisymmetrization_ factor 
        fact *= (1 - ((-1)**(self.T + self.S_ket)))
        
        if self.isNullValue(fact):
            return 0.0
        fact *= ( (2*self.bra.l1 + 1) * (2*self.bra.l2 + 1)
                 *(2*self.ket.l1 + 1) * (2*self.ket.l2 + 1))**0.5
        
        fact *= safe_3j_symbols(self.bra.l1, self.L_bra, self.bra.l2, 0, 0, 0)
        fact *= safe_3j_symbols(self.ket.l1, self.L_ket, self.ket.l2, 0, 0, 0)
        fact *= self.PARAMS_FORCE[DensityDependentParameters.constant]/(4*np.pi)
        
        if self.isNullValue(fact):
            return 0.0
        
        _A = self.PARAMS_SHO.get(SHO_Parameters.A_Mass)
        _Z = self.PARAMS_SHO.get(SHO_Parameters.Z)
        fa = atrE.ForceArgs
        if ((fa.DensDep.protons in self.PARAMS_CORE)
            and (fa.DensDep.neutrons in self.PARAMS_CORE)
            and (self.PARAMS_CORE[fa.DensDep.protons]  > 0)
            and (self.PARAMS_CORE[fa.DensDep.neutrons] > 0)):
            ## it only reset the DD core if there both parameters are defined 
            ## and non zero
            _Z = self.PARAMS_CORE[fa.DensDep.protons]
            _A =  _Z + self.PARAMS_CORE[fa.DensDep.neutrons]
        
        args = (
            QN_1body_radial(self.bra.n1, self.bra.l1), 
            QN_1body_radial(self.bra.n2, self.bra.l2),
            QN_1body_radial(self.ket.n1, self.ket.l1),
            QN_1body_radial(self.ket.n2, self.ket.l2),
            self.PARAMS_SHO.get(SHO_Parameters.b_length), _A, _Z,
            self.PARAMS_FORCE.get(DensityDependentParameters.alpha),
            self.PARAMS_CORE .get(AttributeArgs.ForceArgs.DensDep.core_b_len)
        )
        if self.DEBUG_MODE:
            _RadialDensityDependentFermi.DEBUG_MODE = True
            
        _RadialDensityDependentFermi._DENSITY_APROX = False
        _RadialDensityDependentFermi._R_DIM = self._R_DIM
            
        radial = _RadialDensityDependentFermi.integral(*args)
        
        return fact * radial


class DensityDependentForceFromFile_JScheme(_TwoBodyMatrixElement_Antisym_JCoupled,
                                            DensityDependentForce_JTScheme):
    
    """
    Force for the D1S integrated over a real Mean field density from Taurus:
        * final_wf.bin
        * get from txt (easy)
    Uses the same parameters as the D1S:
        TODO: Might be necessary the Levedeb_ integration or some sort of 
            importing for a very precise grid (i,e, Omega=20)
        # Laguerre_ is also valid x4 points
        
        
        # use J scheme to integrate directly over the j-m functions,
    
    """
    
    COUPLING        = CouplingSchemeEnum.JJ
    RECOUPLES_LS    = False
    _BREAK_ISOSPIN  = True
    
    _R_DIM = 20
    _A_DIM =  0
    _OMEGA = 20
    USING_LEBEDEV = True
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        super(DensityDependentForceFromFile_JScheme, cls).\
            setInteractionParameters(*args, **kwargs)
            
        # Get the filename
        cls._file2import    = None
        cls._density_matrix = None
        cls._kappa_matrix   = None
        cls._sh_states      = [] # <int> index in l_ge_10 format
        cls._sp_states      = [] # <QN_1body_jj> with m and mt
        cls._sp_dim         = None
        cls._orbital_max_sph_harmonic = 0
        
        cls._r   = []  # r of the grid for the me, = HO_b*(x/Alpha+2)^1/2
        cls._ang = []
        cls._weight_r   = []
        cls._weight_ang = []
        cls._spatial_dens = None
        cls._spatial_dens_alp = None
        
        cls._radial_2b_wf_memo  = {}
        cls._sph_harmonic_memo  = {}
        
        ## Define the ANGULAR and RADIAL grid.
        cls.setIntegrationGrid(cls._R_DIM, cls._OMEGA)
        ## Set up the densities from Taurus_vap File 
        cls._importDensityMatrixWFAndSetUpBasis()
        ## Evaluate the spatial-density (integral points must be studied for the case)
        cls.evalutate_spatial_density()
        
        ## Reset the states to use in the new valence space.
        cls._sh_states      = [] # <int> index in l_ge_10 format
        cls._sp_states      = [] # <QN_1body_jj> with m and mt
        cls._sp_dim         = 0
        cls._orbital_max_sph_harmonic = 0
    
    @classmethod
    def setIntegrationGrid(cls, R_dim, OmegaOrd):
        """
        Reset the grid for integration for Laguerre-Associated (radial) 
        and Legendre (Angular).
            Implement Levedev Integral (At least one for a large Omega order)
        """
        if SCIPY_INSTALLED:
            from scipy.special import roots_genlaguerre, roots_legendre
        else:
            raise MatrixElementException("Scipy Required to use this MatrixElement")
        B_LEN  = cls.PARAMS_SHO.get(SHO_Parameters.b_length, 1.0)
        ALPHA_ = cls.PARAMS_FORCE.get(dd_p.alpha,  1.0)
        ## Radial Grid
        xR, wR, sum_ = roots_genlaguerre(R_dim, 0.5, True)
        cls._r =  B_LEN * np.sqrt(xR / (2.0 + ALPHA_)) # 
        cls._weight_r = wR
        
        cls._R_DIM = R_dim
        cls._OMEGA = OmegaOrd
        ## Angular Grid: Choice between stored Lebedev_ grid mesh (commonly used).
        cls._ang = []
        cls._weight_ang = []
        if cls.USING_LEBEDEV:
            ## Angular Grid from Lebedev_ Imported from /docs/LebedevPointsWeights
            if cls._OMEGA % 2 == 1:
                cls._OMEGA += 1
            if cls._OMEGA < 14: 
                print(" WARNING: Minimum Omega for Lebedev is 14, fixing to that.")
                cls._OMEGA = 14
            elif cls._OMEGA > 22:
                print(" WARNING: Maximum Omega for Lebedev is 22, fixing to that.")
                cls._OMEGA = 22
            ## read the file and convert the numbers
            with open(f'docs/LebedevPointsWeightsOmg{cls._OMEGA}.gut', 'r') as f:
                for line in f.readlines()[1:]:
                    line = line.strip().replace('D', 'E')
                    i, costh, phi, wA = line.split()
                    cls._ang.append( (float(costh), float(phi)) )
                    cls._weight_ang.append( float(wA) )
                    cls._A_DIM += 1
        else:
            ## Angular Grid (Theta_dim=Phi_dim = Ang_dim)
            xA, wA, sum_ = roots_legendre(OmegaOrd, True)
            costh = xA
            phi   = np.pi * copy(xA)
            for i in range(OmegaOrd):
                for j in range(OmegaOrd):
                    cls._ang.append( (costh[i], phi[j]) )
                    cls._weight_ang.append(wA[i] * wA[j])
            cls._A_DIM = cls._OMEGA**2       
    
    @classmethod
    def setBasis2BFunctions(cls, HO_b_length):
        """
        Once given the w.f. and having the b_lengths (core, m.e.)
        Set the angular and Radial functions and angular coefficents.
        """
                        
        radial2b = _buildRadialFunctionsArray(cls._sh_states, HO_b_length)
        cls._radial_2b_wf_memo = deepcopy(radial2b)
        
        # Build Base for Angular and Radial coefficients of the quantum numbers
        K_max = _buildAngularYCoeffsArray(cls._sh_states)
        cls._orbital_max_sph_harmonic = K_max
        
        # from scipy.special import sph_harm
        ## Angular basis (Common for both cases)       
        for K in range(cls._orbital_max_sph_harmonic +1):
            for M in range(-K, K +1):
                ind_k = angular_Y_KM_index(K, M, False)
                
                ang_func = [sphericalHarmonic(K, M, angle) for angle in cls._ang]
                ang_func = np.array(ang_func)
                cls._sph_harmonic_memo[ind_k] = deepcopy(ang_func)
    
    def _actualizeTheBasisFunctions(self):
        """ 
        Run each time the introduction of a new sh state, to be run at the
        constructor.
        """
        
        new_found = False      
        for st_ in (self.bra.sp_state_1.AntoineStrIndex_l_greatThan10, 
                    self.bra.sp_state_2.AntoineStrIndex_l_greatThan10, 
                    self.ket.sp_state_1.AntoineStrIndex_l_greatThan10, 
                    self.ket.sp_state_2.AntoineStrIndex_l_greatThan10):
            
            if st_ not in self._sh_states:
                self._sh_states.append(int(st_))
                n, l, j = readAntoine(st_, l_ge_10=True)
                self._sp_dim += 2*(j + 1) # counting p and n
                new_found = True
        
        if new_found:
            self._sp_states = []
            for mt in (-1, 1):
                for sh_st in self._sh_states:
                    n, l, j = readAntoine(sh_st, l_ge_10=True)
                    m_vals = [i for i in range(j, -j-1, -2)]
                    for m in m_vals:
                        self._sp_states.append(QN_1body_jj(n, l, j, m, mt))
            
            self.setBasis2BFunctions(self.PARAMS_SHO[SHO_Parameters.b_length])
        
    @classmethod
    def _angularComponentsForDensity(cls, a,la,ja,mja,indx_a, b,lb,jb,mjb,indx_b):
        """ Find the K,M components of the two body angular wave function and
        add up to the spatial density 
        
        Note, in order to speed up the density matrix summation"""
        spO2 = cls._sp_dim // 2
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
                    
                    val_p = val * cls._density_matrix[b, a]
                    val_n = val * cls._density_matrix[b + spO2, a + spO2]
                    
                    val = val_p + val_n
                    cls._spatial_dens_imag[ir, ia] += np.imag(val)
                    cls._spatial_dens[ir, ia]      += np.real(val)
    
    @classmethod
    def evalutate_spatial_density(cls):
        """
        Evaluate the radial and angular functions for the Mean field W.F.
        """
        global _angular_Y_KM_me_memo
        import time
        t_ = time.time()
        print(f" [ ] Evaluating Spatial Density ...")
        
        cls._spatial_dens = np.zeros( (cls._R_DIM, cls._A_DIM) )
        cls._spatial_dens_imag = np.zeros( (cls._R_DIM, cls._A_DIM) ) ## test
        cls._spatial_dens_alp  = np.zeros( (cls._R_DIM, cls._A_DIM) )
        tot_ = (cls._sp_dim // 2) * (cls._sp_dim // 2 + 1) // 2
        for a  in range(cls._sp_dim // 2):
            # print("  progr. a% =", tot_*(1 - a) - (a*(a-1)//2), "/", tot_)
            print("  progr a:", a+1, "/", cls._sp_dim // 2)
            sp_st_a = cls._sp_states[a]
            na, la, ja, mja = sp_st_a.n, sp_st_a.l, sp_st_a.j, sp_st_a.m
            indx_a = angular_Y_KM_index(ja, mja, True)
            
            for b in range(a, cls._sp_dim // 2):
                sp_st_b = cls._sp_states[b]
                
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
        
        integ_A = 0.0
        ALPHA_  = cls.PARAMS_FORCE[dd_p.alpha]
        for ir in range(len(cls._r)):
            radial  = np.exp(( (2.0+ALPHA_) * (cls._r[ir] / cls._b_density)**2))
            for ia in range(len(cls._ang)):
                val  = cls._spatial_dens[ir, ia] 
                val *= cls._weight_ang[ia] * cls._weight_r[ir]
                integ_A +=  radial * val
                
                cls._spatial_dens_alp[ir,ia] = cls._spatial_dens[ir,ia] **ALPHA_
        
        integ_A *= (cls._b_density**3) / (2.0*((2.0 + ALPHA_)**1.5))
        if cls.USING_LEBEDEV:
            integ_A *= 4 * np.pi
        else:
            integ_A *= np.pi ## not checked
        _= 0
        
        import matplotlib.pyplot as plt
        #
        # fig_ = plt.figure()
        # for ia in range(0, len(cls._ang), cls._OMEGA):
        #     lab = f'ang={cls._ang[ia][0]:4.2f}, {cls._ang[ia][1]:4.2f}'
        #     plt.plot(cls._r, cls._spatial_dens[:, ia], label=lab)
        #
        # plt.legend()
        # plt.show()
        #
        fig_ = plt.figure
        off_diag = deepcopy(cls._density_matrix)
        # for i in range(cls._sp_dim):
        #     if off_diag[i,i] > 0.1:
        #         off_diag[i,i] = 0
        plt.imshow(off_diag)
        plt.show()
        # TODO:: Test the density
        t_ = (time.time() - t_)
        print( " [DONE] Spatial Density has been imported and evaluated. ",
              f"({t_:5.3f}s) A={integ_A:9.5f}")
                
        
    
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
        
        cls._sp_dim = dim_
        cls._density_matrix = np.matmul(V.conjugate(), np.transpose(V)) 
        cls._kappa_matrix   = np.matmul(V.conjugate(), np.transpose(U))
        
        # print("Warning, copying 16O diagonal wf")
        # cls._density_matrix = np.zeros((dim_, dim_))
        # for i in range(dim_ // 2):
        #     if i >= 8: 
        #         continue
        #     cls._density_matrix[i,i] = 1
        #     cls._density_matrix[(dim_//2) + i,(dim_//2) + i] = 1
        
        ## Set the sp_states in the file order
        for mt in (-1, 1):
            for sh_st in cls._sh_states:
                n, l, j = readAntoine(sh_st, l_ge_10=True)
                m_vals = [i for i in range(j, -j-1, -2)]
                for m in m_vals:
                    cls._sp_states.append(QN_1body_jj(n, l, j, m, mt))
        
    @classmethod
    def _readBinary(cls):
        """
        Equivalent process to import the file as the txt.
        
        * Tested against the same WF, exported both txt/bin, for several basis
        """
        sh_dim = 0
        density_matrix_lines = []
        
        x = np.fromfile(cls._file2import, count=2, dtype='uint32')
        sh_dim = int(x[-1])
        x = np.fromfile(cls._file2import, count=sh_dim +4, dtype='uint32')
        cls._sh_states = list(x[4:])
        
        y = np.fromfile(cls._file2import, dtype=np.float64)
        y = y[sh_dim+2:] # counting the lines=[sh_dim, lines sh, label long]
        
        dim_   = (y.size - 1) // 2 #! there is an auxiliary line between U and V
        u0_mat = y[:dim_]
        v0_mat = y[dim_+1:]        #! there is an auxiliary line between U and V
        
        ## NOTE: verified with the txt file for the same function
        cls._setUpDensityMatrices(dim_, u0_mat, v0_mat)
    
    @classmethod
    def _readTXT(cls):
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
        with open(cls._file2import, 'r') as f:
            data = f.readlines()
            for il, line in enumerate(data):
                line = line.strip()
                if il == 0:
                    sh_dim = int(line)
                elif il <= sh_dim:
                    cls._sh_states.append(int(line))
                elif il == sh_dim + 1:
                    continue
                else:
                    # if 'E' in line:
                    #     line = line.replace('E', 'e')
                    line = float(line)
                    density_matrix_lines.append(line)
        
        dim_ = len(density_matrix_lines) // 2
        u0_mat = density_matrix_lines[:dim_]
        v0_mat = density_matrix_lines[dim_:]
        
        ## NOTE: verified with the DIMENS_indexes_and_rhoLRkappas.gut file
        cls._setUpDensityMatrices(dim_, u0_mat, v0_mat)
            
    @classmethod
    def _importDensityMatrixWFAndSetUpBasis(cls):
        """ 
        Read the file for the Us and Vs and convert to density matrix.
        Shell states are also in the file, set up the radial/angular basis for
        the imported density. (radial has to be reset after)
        """
        cls._file2import = cls.PARAMS_FORCE.get(dd_p.file, None)
        if cls._file2import.endswith('.bin'):
            ## import binary file
            cls._readBinary()
        elif cls._file2import.endswith('.txt'):
            ## import txt file
            cls._readTXT()
        else:
            raise MatrixElementException(f"Invalid File to import [{cls._file2import}]")
        
        b_length = cls.PARAMS_SHO.get(SHO_Parameters.b_length, 1.0)
        ## Use the b_length for the calculation as default.
        if (dd_p.core in cls.PARAMS_FORCE):
            ## In case of b_core explicitly given, use it for the w.f.
            _b_core = cls.PARAMS_FORCE[dd_p.core].get(atrE.ForceArgs.DensDep.core_b_len)
            if ( (_b_core != None) and (_b_core.replace('.', '').isnumeric()) ):
                b_length = float(_b_core)
        cls._b_density = b_length
        
        ## Do the wave function setting for the density
        cls.setBasis2BFunctions(b_length)

    def __checkInputArguments(self, bra, ket):
        if not isinstance(bra, QN_2body_jj_J_Coupling):
            raise MatrixElementException("<bra| is not <QN_2body_jj_J_Coupling>")
        if not isinstance(ket, QN_2body_jj_J_Coupling):
            raise MatrixElementException("|ket> is not <QN_2body_jj_J_Coupling>")
    
    def __init__(self, bra, ket, run_it=True):
        
        self.__checkInputArguments(bra, ket)
        
        self.bra = bra
        self.ket = ket
        
        self.J = bra.J
        
        self.exchange_phase = None
        self.exch_2bme = None
        
        if (bra.sp_state_1.m_t + bra.sp_state_2.m_t != 
            ket.sp_state_1.m_t + ket.sp_state_2.m_t):
            self._value = 0.0
        else:
            self._nullConditionForSameOrbit()
        
        if not self.isNullMatrixElement and run_it: # always run it
            self._run()
              
    def _run(self):
        
        if self.isNullMatrixElement:
            return
        
        self._actualizeTheBasisFunctions()
        
        if self.DEBUG_MODE: 
            XLog.write('nas_me', ket=self.ket.shellStatesNotation)
        
        # antisymmetrization_ taken in the inner evaluation            
        self._calculateJMatrixElement()
        
        if self.DEBUG_MODE:
            XLog.write('nas_me', value=self._value, norms=self.bra.norm()*self.ket.norm())
        
        self._value *= self.bra.norm() * self.ket.norm()
    
            
    def _calculateJMatrixElement(self):
        """
        Process has the antisymmetrization in the uncoupled matrix element.
        Process does not require the LS coupling
        """        
        
        ALPHA_ = self.PARAMS_FORCE[dd_p.alpha]
        B_LEN_ = self.PARAMS_SHO[SHO_Parameters.b_length]

        na, la, ja = self.bra.n1, self.bra.l1, self.bra.j1
        nb, lb, jb = self.bra.n2, self.bra.l2, self.bra.j2
        nc, lc, jc = self.ket.n1, self.ket.l1, self.ket.j1
        nd, ld, jd = self.ket.n2, self.ket.l2, self.ket.j2
        
        nl_ab = (na,la, nb,lb)
        nl_cd = (nc,lc, nd,ld)    
        
        radial  = self._radial_2b_wf_memo[nl_ab](self._r)
        radial *= self._radial_2b_wf_memo[nl_cd](self._r)
        radial *= self._weight_r * np.exp((2.+ALPHA_)* np.power(self._r/B_LEN_, 2))
        self._curr_radial = radial
        
        ma_valid = [m for m in range(-self.bra.j1, self.bra.j1 +1, 2)]
        mb_valid = [m for m in range(-self.bra.j2, self.bra.j2 +1, 2)]
        
        mc_valid = [m for m in range(-self.ket.j1, self.ket.j1 +1, 2)]
        md_valid = [m for m in range(-self.ket.j2, self.ket.j2 +1, 2)]
        
        me_value = 0.0
        for ma in ma_valid:
            for mb in mb_valid:
                if self.bra.M != (ma+mb)//2: continue
                
                args_b = (S(ja)/2, S(jb)/2, S(self.bra.J),
                          S(ma)/2, S(mb)/2, S(self.bra.M))
                ccg_b = safe_clebsch_gordan(*args_b)
                if self.isNullValue(ccg_b): continue
                
                for mc in mc_valid:
                    for md in md_valid:
                        
                        if self.ket.M != (mc+md)//2: continue
                        args_k = (S(jc)/2, S(jd)/2, S(self.ket.J),
                                  S(mc)/2, S(md)/2, S(self.ket.M))
                        ccg_k = safe_clebsch_gordan(*args_k)
                        ang_recoup = ccg_b * ccg_k
                        if self.DEBUG_MODE:
                            XLog.write('junc', ccg_b=ccg_b, ccg_k=ccg_k,
                                       ma=ma,mb=mb,mc=mc, md=md)
                        if self.isNullValue(ang_recoup): continue
                        
                        dd_integral = self._radialAngularIntegral(ma,mb,mc,md)
                        me_value += (ang_recoup * dd_integral)
                        
                        if self.DEBUG_MODE: XLog.write('junc', dd_int=dd_integral)
        
        self._value = me_value *  self.PARAMS_FORCE[dd_p.constant]
    
    def _isospinExchangeFactor(self):
        """ Matrix element factor associated with the isospin """
        ta,tb = self.bra.sp_state_1.m_t, self.bra.sp_state_2.m_t
        tc,td = self.ket.sp_state_1.m_t, self.ket.sp_state_2.m_t
        X_ac_bd  = ((ta==tc)*(tb==td))
        X_ac_bd -= self.PARAMS_FORCE[dd_p.x0]*((ta==td)*(tb==tc))
        X_ad_bc  = ((ta==td)*(tb==tc)) 
        X_ad_bc -= self.PARAMS_FORCE[dd_p.x0]*((ta==tc)*(tb==td))
        
        self.X_ac_bd = X_ac_bd
        self.X_ad_bc = X_ad_bc            
                 
    def _radialAngularIntegral(self, mja, mjb, mjc, mjd):
        
        self._isospinExchangeFactor()
        if abs(self.X_ac_bd - self.X_ad_bc) < 1.0e-6: 
            return 0.0
        
        ALPHA_ = self.PARAMS_FORCE[dd_p.alpha]
        B_LEN_ = self.PARAMS_SHO[SHO_Parameters.b_length]
        # for ir in range(len(cls._r)):
        la, ja = self.bra.l1, self.bra.j1
        lb, jb = self.bra.l2, self.bra.j2
        lc, jc = self.ket.l1, self.ket.j1
        ld, jd = self.ket.l2, self.ket.j2
        
        indx_a = angular_Y_KM_index(ja, mja, True)        
        indx_b = angular_Y_KM_index(jb, mjb, True)
        indx_c = angular_Y_KM_index(jc, mjc, True)        
        indx_d = angular_Y_KM_index(jd, mjd, True)
        
        # nl_ab = (na,la, nb,lb)
        # nl_cd = (nc,lc, nd,ld)    
        #
        # radial  = self._radial_2b_wf_memo[nl_ab](self._r)
        # radial *= self._radial_2b_wf_memo[nl_cd](self._r)
        # radial *= self._weight_r * np.exp((2.+ALPHA_)* np.power(self._r/B_LEN_, 2))
        
        aux_d   = np.zeros(self._A_DIM)
        aux_e   = np.zeros(self._A_DIM)
        ## DIR
        if not self.isNullValue(self.X_ac_bd):
            if self.DEBUG_MODE:  XLog.write('rangDir')
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
                    
                    if self.DEBUG_MODE:
                        XLog.write('rangDir', km1=(K1,M1), km2=(K2,M2), cang=c_ang)
        
        ## EXCH
        if not self.isNullValue(self.X_ad_bc):
            if self.DEBUG_MODE:  XLog.write('rangExc')
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
                    if self.DEBUG_MODE:
                        XLog.write('rangExc', km1=(K1,M1), km2=(K2,M2), cang=c_ang)
        
        angular = (self.X_ac_bd * aux_d) - (self.X_ad_bc * aux_e)
        angular = angular * self._weight_ang
        
        me_val = 0.0
        # for ir in range(len(self._r)):
        #     for ia in range(len(self._ang)):
        #         val = self._curr_radial[ir] * angular [ia]
        #         me_val += val * (self._spatial_dens_alp[ir,ia])
        
        ## Same loop but using a the matrix product  [rad]*[dens_pow]*[ang]
        me_val = np.inner(self._curr_radial, 
                          np.inner(self._spatial_dens_alp, angular))
        
        me_val *= 0.5 * (B_LEN_ **3)
        me_val /= (2.0 + ALPHA_)**1.5
        if self.USING_LEBEDEV:
            me_val *= 4 * np.pi
        else:
            me_val *= np.pi ## not checked
        if almostEqual(me_val, 0, 1.0e-6):
            return 0.0
        if abs(np.imag(me_val)) > 1.0e-10:
            raise MatrixElementException(f"DD term is complex <{self.bra} |v| {self.ket}>= {me_val}") 
        return np.real(me_val)
    
    ## return void LS valid L S for SpeedRunner to work with this m.e
    def _validKetTotalSpins(self):
        return tuple()
    
    def _validKetTotalAngularMomentums(self):
        return tuple()




class Density(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        
