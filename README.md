# 2B_MatrixElements
This is a research project with the goal of implementing and evaluate different two-body matrix elements. The main base for the computation is the **spherical harmonic oscillator** and the matrix elements are coupled in *J* or *JT* schemes.

## Installation
Clone or download the repository. The package requirements are:
1. Python 3, at least version 3.8 due the internal way of usage of dictionaries and functions arguments.
2. numpy, scipy & sympy
3. pandas     (for testing)
4. matplotlib (for testing)

I.e using ```pip`` for the libraries:

```
pip install pandas
```

To update Python into a newer version see i.e. [https://tech.serhatteker.com/post/2019-12/upgrade-python38-on-ubuntu/](How to Upgrade to Python 3.8 on Ubuntu)

## Theory
Many nuclear codes require require the implementation of a certain Hamiltonian two body interaction, for example, *Interaction-Shell-model* solves the Schrödinger equation by direct diagonalization of these hamiltonians. Others, like *Mean Field-HFB* like codes implement internally (predefined) the two-body Hamiltonian to access large valence spaces

In this suite, we develop the tools to implement them and program several commonly use potentials. The main base of single-particle wave functions is the *spherical harmonic oscillator SHO* (but could be extended to another bases).

The suite have extensions to connect with **taurus_vap** code, also for the different types of Hamiltonians and for the *center of mass* correction of this program for *no core* calculations. See details of that code in:

[![DOI](https://doi.org/10.1140/epja/s10050-021-00369-z)](Symmetry-projected variational calculations with the numerical suite TAURUS)

Interactions are deal with the *object oriented paradigm*, in order to deal with their hierarchical complexity (coupling schemes, radial implementation, wave functions or specific constructions) and reusage of certain process. Details of these ideas and the parameters/definitions of currently implemented matrix elements are in [https://github.com/migueldelafuente1/2B_MatrixElements/blob/main/docs/How_to_2B_MatrixElements.pdf](docs/How_to_2B_MatrixElements.pdf).

## Usage: 
### Compute a whole valence space using TBME_Runner, m.e format à la Antoine

This program was originally designed to feed the interactions for a nuclear mean field code ([taurus_vap](https://github.com/project-taurus/taurus_vap)), that solves the Hartree-Fock-Bogoliubov equations for an arbitrary two-body interaction. The requirements for the Hamiltonian files are based in another shell model code called [Antoine](http://www.iphc.cnrs.fr/nutheo/code_antoine/menu.html), here we use its notation for the harmonic oscillator base.

The requirements for **taurus_vap** Hamiltonians are explained in *docs/How_to_2B_MatrixElements*. The class to do this computation is called **TBME_Runner** and it can only be set by the input file (see How to in the previous document).

```python {cmd}
from helpers import TBME_Runner

# Write an xml input: define HBarOmega, output filename, Forces ...

# Read and run the 
computation = TBME_Runner('my_input_file.xml')
computation.run()
 
# >> Computes and save the results (creates a /result folder) 
```

Currently implemented two body interactions (All computed in JT scheme):
1. **Central** interactions in the form of gaussians, r powers, 1/r or yukawa.
2. **Coulomb** interaction (J scheme)
3. **Brink-Boeker** like interactions or gaussian potential series.
4. **Short range Spin-Orbit** approximation potential
5. Fixed **Density dependent** (Nuclear Fermi shell filling approximation).
6. **Series of gaussians**, an extension of the Brink-Boeker sum of gaussians to as many you want (To expand another potentials).
7. Get them **from a file**, reuse previous computations to save time (also valid to multiply the results by a global factor).
8. **Kinetic two-body** matrix element, necessary to evaluate the *center of mass* correction. Be careful with the internal setting of nucleon mass and *HbarC* constants in the code.

### Evaluating matrix elements individually

For educational, testing and other purposes, the elements are instanciable to be evaluated. The progress or steps can be printed or debbuged with suitable tools (See for example the ``XLog`` class for debugging the matrix element in tree format).

The interactions are classes, where the interaction parameters will be class attributes (statics). The matrix elements are therefore instances of an interaction class. The computation is done internally and then you get the value as an object ``@property``.

```python {cmd}
from helpers.TBME_Runner import TBME_Runner
from helpers.WaveFunctions import QN_1body_jj, QN_2body_jj_JT_Coupling
from matrix_elements.CentralForces import CentralForce_JTScheme
from helpers.Enums import PotentialForms, SHO_Parameters, BrinkBoekerParameters

# 1. Define the parameters (Names standardized by enumerations) 
kwargs = {
    SHO_Parameters.A_Mass       : 4,
    SHO_Parameters.b_length     : 1.4989,
    SHO_Parameters.hbar_omega   : 18.4586,
    CentralMEParameters.potential : PotentialForms.Power,
    CentralMEParameters.mu_length : 1.0,
    CentralMEParameters.constant  : 1.0,
    CentralMEParameters.n_power   : 0,
}

# 2. Class setting (i.e, set parameters and turn on the XLog debugging)
CentralForce_JTScheme.setInteractionParameters(**kwargs)
CentralForce_JTScheme.turnDebugMode(True)

# 3. Instance the class for the matrix element <0s1/2 0s1/2|V|0s1/2 0s1/2 (J:3T:1)>
#    evaluation is done internally
me = CentralForce_JTScheme(
    	QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1), QN_1body_jj(0,0,1),1, 0),
    	QN_2body_jj_JT_Coupling(QN_1body_jj(0,0,1), QN_1body_jj(0,0,1),1, 0),
    )
    
# 4. get the m.e. value attribute and save the Log tree (Recoupling, Moshinsky series, etc) in a xml file.
print("me: ", me.value)
me.saveXLog('me_test')
```


### Efficient computing of valence Spaces. TBME_SpeedRunner.
Currently developing ...

**TBME_Runner** was a first approach to evaluate the matrix elements, but evaluates the interactions one by one. That is: set an interaction, run the whole valence space for it and then repeat with the next. Nevertheless, most of the matrix elements, besides being in *J* or *JT scheme*, they internally are evaluated in the *LS scheme*. The new runner (**TBME_SpeedRunne**) invert the process: first it set each interaction (cannot be of the same type) and then evaluates the valence space, doing all the LS/LST evaluations in an intermediate step.