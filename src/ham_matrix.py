from rdkit import Chem
from Molecule import Molecule
import numpy as np
import jax
import jax.numpy as jnp

def findOrbitalNeighbor(vo_list, neighbor_atom_idx, double_bond):
    for index, vo in enumerate(vo_list):
        if vo.atom.idx == neighbor_atom_idx:
            if double_bond == True:
                return index + 1
            else:
                return index
    return -1

def findParameter(parameter_matrix, atom1, orbital1, atom2, orbital2):
    element_list = ["H", "C", "N", "O", "S", "Cl"]
    orbital_list = ["P", "S", "SP", "SP2", "SP3"]
    if atom2 == None:
        v2 = len(element_list)
        v4 = 0
    else:
        v2 = element_list.index(atom2)
        v4 = orbital_list.index(orbital2)
    return parameter_matrix[element_list.index(atom1)][v2][orbital_list.index(orbital1)][v4]

def findParameterIndex(atom1, orbital1, atom2, orbital2, element_list, orbital_list):
    if atom2 == None:
        v2 = len(element_list)
        v4 = 0
    else:
        v2 = element_list.index(atom2)
        v4 = orbital_list.index(orbital2)
    return(element_list.index(atom1), v2, orbital_list.index(orbital1), v4)

def grabIndices(smi, element_list, orbital_list):
    mol = Molecule(smi)
    m_init = Chem.MolFromSmiles(smi)
    m = Chem.AddHs(m_init)
    vo_list = mol.get_all_valence_orbitals()
    atom_list = m.GetAtoms()
    double_bond = [False] * len(atom_list)
    prev_atom_index = 0
    all_indices = []
    for index, vo in enumerate(vo_list):
        # Check atom index with previous and reset double bond list as necessary
        atom_index = vo.atom.idx
        if atom_index != prev_atom_index:
            double_bond = [0] * len(atom_list)

        # Find orbital neighbor index and add to double bond list
        if vo.neighbor != None:
            neighbor_vo_idx = findOrbitalNeighbor(vo_list, vo.neighbor.atom.idx, double_bond[vo.neighbor.atom.idx])
            double_bond[vo.neighbor.atom.idx] += 1
    
        # Grab and fix hybridization of alpha parameter
        hybrid = atom_list[atom_index].GetHybridization()
        if str(hybrid) == "UNSPECIFIED":
            hybrid = "S"
        elif vo.neighbor != None and double_bond[vo.neighbor.atom.idx] == 2:
            hybrid = "P"
    
        # Add alpha hamiltonian and parameter index to list
        all_indices.append([(index, index), findParameterIndex(str(vo.atom.atom_type), str(hybrid), None, None, element_list, orbital_list)])

        # Grab and fix hybridization of neighbor for beta parameter
        if vo.neighbor != None:
            neighbor_hybrid = atom_list[vo.neighbor.atom.idx].GetHybridization()
            if str(neighbor_hybrid) == "UNSPECIFIED":
                neighbor_hybrid = "S"
            elif double_bond[vo.neighbor.atom.idx] == 2:
                neighbor_hybrid = "P"

            # Add beta hamiltonian and parameter index to list
            all_indices.append([(index, neighbor_vo_idx), findParameterIndex(str(vo.atom.atom_type), str(hybrid), str(vo.neighbor.atom.atom_type), str(neighbor_hybrid), element_list, orbital_list)])

        # Set previous atom index to check whether the next atom is a new one
        prev_atom_index = vo.atom.idx
    return all_indices, len(vo_list)

def buildHamiltonian(smi, parameter_matrix):
    mol = Molecule(smi)
    m_init = Chem.MolFromSmiles(smi)
    m = Chem.AddHs(m_init)
    vo_list = mol.get_all_valence_orbitals()
    atom_list = m.GetAtoms()
    hamiltonian = [[0 for col in range(len(vo_list))] for row in range(len(vo_list))]
    double_bond = [False] * len(atom_list)
    prev_atom_index = 0
    for index, vo in enumerate(vo_list):
        # Check atom index with previous and reset double bond list as necessary
        atom_index = vo.atom.idx
        if atom_index != prev_atom_index:
            double_bond = [0] * len(atom_list)

        # Find orbital neighbor index and add to double bond list
        if vo.neighbor != None:
            neighbor_vo_idx = findOrbitalNeighbor(vo_list, vo.neighbor.atom.idx, double_bond[vo.neighbor.atom.idx])
            double_bond[vo.neighbor.atom.idx] += 1
    
        # Grab and fix hybridization of alpha parameter
        hybrid = atom_list[atom_index].GetHybridization()
        if str(hybrid) == "UNSPECIFIED":
            hybrid = "S"
        elif vo.neighbor != None and double_bond[vo.neighbor.atom.idx] == 2:
            hybrid = "P"
    
        # Place alpha value in Hamiltonian
        hamiltonian[index][index] = findParameter(parameter_matrix, str(vo.atom.atom_type), str(hybrid), None, None)

        # Grab and fix hybridization of neighbor for beta parameter
        if vo.neighbor != None:
            neighbor_hybrid = atom_list[vo.neighbor.atom.idx].GetHybridization()
            if str(neighbor_hybrid) == "UNSPECIFIED":
                neighbor_hybrid = "S"
            elif double_bond[vo.neighbor.atom.idx] == 2:
                neighbor_hybrid = "P"

            # Place beta value in Hamiltonian
            hamiltonian[index][neighbor_vo_idx] = findParameter(parameter_matrix, str(vo.atom.atom_type), str(hybrid), str(vo.neighbor.atom.atom_type), str(neighbor_hybrid))

        # Set previous atom index to check whether the next atom is a new one
        prev_atom_index = vo.atom.idx
    return hamiltonian

def calculateEnergy(smi, parameter_matrix):
    eigenval = np.linalg.eigvals(np.array(buildHamiltonian(smi, parameter_matrix)))
    eigenval = np.sort(eigenval)
    energy = 0
    for i in range(int(len(eigenval)/2)):
        energy += eigenval[i]
    return energy

def calculateHOMOLUMO(smi, parameter_matrix):
    eigenval = jax.numpy.linalg.eigvals(jax.numpy.array(buildHamiltonian(smi, parameter_matrix)))
    eigenval = jax.numpy.sort(eigenval)
    length = len(eigenval)
    mid = round(length/2)
    ahhh = jax.numpy.concatenate((jax.numpy.zeros(mid - 1), jax.numpy.ones(2), jax.numpy.zeros(length - mid - 1)))
    gap = jax.numpy.sum(jax.numpy.dot(eigenval, ahhh))
    print(gap)
    return gap

def _get_multiplicity(n_electrons: int):
    """Multiplicity
    Args:
        n_electrons (int): number of electrons
    Returns:
        Any: multiplicity
    """
    return (n_electrons % 2) + 1


def set_occupations(electrons, energies, n_orbitals):
    """Occupation
    Args:
        electrons (int): number of electrons
        energies (Any): Hückel's eigenvalues
        n_orbitals (int): number of orbitals
    Returns:
        Tuple: occupation, spin occupation, number of occupied orbitals, number of unpair electrons
    """
    charge = 0
    n_dec_degen = 3
    n_electrons = electrons - charge
    multiplicity = _get_multiplicity(n_electrons)
    n_excess_spin = multiplicity - 1

    # Determine number of singly and doubly occupied orbitals.
    n_doubly = int((n_electrons - n_excess_spin) / 2)
    n_singly = n_excess_spin
    # Make list of electrons to distribute in orbitals
    all_electrons = [2] * n_doubly + [1] * int(n_singly)
    # Set up occupation numbers
    occupations = np.zeros(n_orbitals, dtype=np.int32)
    # Loop over unique rounded orbital energies and degeneracies and fill with
    # electrons
    energies_rounded = energies.round(n_dec_degen)
    unique_energies, degeneracies = np.unique(
        energies_rounded, return_counts=True)
    for energy, degeneracy in zip(np.flip(unique_energies), np.flip(degeneracies)):
        if len(all_electrons) == 0:
            break

        # Determine number of electrons with and without excess spin.
        electrons_ = 0
        for _ in range(degeneracy):
            if len(all_electrons) > 0:
                pop_electrons = all_electrons.pop(0)
                electrons_ += pop_electrons

        # Divide electrons evenly among orbitals
        # occupations[jnp.where(energies_rounded == energy)] += electrons / degeneracy
        occupations[energies_rounded == energy] = electrons_ / degeneracy
    return occupations

def grabOccupations(smi, eigvals, size):
    mol = Molecule(smi)
    electrons = 0
    for atom in mol.atoms:
        electrons += atom.valence_electrons
    occupations = set_occupations(electrons, eigvals, size)
    return occupations
