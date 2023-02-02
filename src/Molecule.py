from typing import Optional, List, Dict
import numpy as np
import logging
import rdkit  # type: ignore
from rdkit import Chem  # type: ignore
#import torch
#from torch_geometric.data import Data
from utils import one_hot_encode, atom_to_num_VOs


sanitizeOps = rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_ALL ^ \
              rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY ^ \
              rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS ^ \
              rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS

class ValenceOrbital:
    def __init__(self, num_electrons: int, atom: 'Atom', neighbor: Optional['ValenceOrbital'] = None):
        self.num_electrons = num_electrons
        self.atom = atom
        self.neighbor = neighbor

    def pair(self, other: 'ValenceOrbital') -> None:
        """Pairs two valence orbitals, setting each neighbor to their respective neighbors
           and num_electrons to -1. Also updates RDKit Mol Objects to reflect new pairings."""
        if self.neighbor is not None or other.neighbor is not None:
            raise ValueError("Cannot pair already paired orbitals!")

        if self.num_electrons + other.num_electrons != 2:
            raise ValueError("Can only pair orbitals with two total electrons!")

        if self.atom is other.atom:
            raise ValueError("Cannot pair an orbital with an orbital on the same atom!")

        current_bond = self.atom.molecule.molecule.GetBondBetweenAtoms(self.atom.idx, other.atom.idx)

        if current_bond is not None and current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
            raise ValueError("Cannot make bond order above Triple!")
            return

        self.neighbor = other
        self.neighbor.neighbor = self

        if self.num_electrons == 0:  # homolytic bond formation; electrons transferred from neighbor
            self.atom.decrement_formal_charge()
            self.neighbor.atom.increment_formal_charge()

        if self.num_electrons == 1:  # heterolytic bond formation; radicals removed
            self.atom.decrement_radical_electrons()
            self.neighbor.atom.decrement_radical_electrons()

        if self.num_electrons == 2:  # homolytic bond formation; electrons transferred to neighbor
            self.atom.increment_formal_charge()
            self.neighbor.atom.decrement_formal_charge()

        self.atom.increment_bond_order(other.atom)

        self.num_electrons = -1
        self.neighbor.num_electrons = -1

    def single_electron_transfer(self, other: 'ValenceOrbital') -> None:
        """Transfers a single electron from self to other (must be empty)"""
        if self.num_electrons != 1 or other.num_electrons != 0:
            raise ValueError("Can only transfer electron from radical to empty orbital!")

        self.num_electrons -= 1
        other.num_electrons += 1
        self.atom.decrement_radical_electrons()
        self.atom.increment_formal_charge()
        other.atom.increment_radical_electrons()
        other.atom.decrement_formal_charge()
        self.atom.molecule.smi = Chem.MolToSmiles(self.atom.molecule.molecule)

    def unpair(self, num_electrons_remaining: int) -> None:
        """Unpairs this valence orbital with its neighbor and sets the 
        number of electrons in the current ValenceOrbital to num_electrons_remaining.
        Also updates RDKit Mol Objects to reflect new pairings."""
        if self.neighbor is None:
            raise ValueError("Cannot unpair an unpaired orbital!")

        if not (0 <= num_electrons_remaining <= 2):
            raise ValueError("num_electrons_remaining must be between 0 and 2!")

        current_bond = self.atom.molecule.molecule.GetBondBetweenAtoms(self.atom.idx, self.neighbor.atom.idx)

        if current_bond is None:
            raise ValueError("Cannot remove non-existent bond!")
            return

        self.num_electrons = num_electrons_remaining
        self.neighbor.num_electrons = 2 - num_electrons_remaining

        if self.num_electrons == 0:  # homolytic bond cleavage; electrons transferred to neighbor
            self.atom.increment_formal_charge()
            self.neighbor.atom.decrement_formal_charge()

        if self.num_electrons == 1:  # heterolytic bond cleavage; radicals formed
            self.atom.increment_radical_electrons()
            self.neighbor.atom.increment_radical_electrons()

        if self.num_electrons == 2:  # homolytic bond cleavage; electrons transferred to neighbor
            self.atom.decrement_formal_charge()
            self.neighbor.atom.increment_formal_charge()

        self.atom.decrement_bond_order(self.neighbor.atom)
        self.neighbor.neighbor = None
        self.neighbor = None

        return True

    def interact_empty_orbital_with_bond(self, other: 'ValenceOrbital') -> None:
        # interacts non-bonding orbital with bonding orbital
        if self.atom is other.atom:
            raise ValueError("Cannot interact an orbital with an orbital on the same atom!")

        if self is other.neighbor:
            raise ValueError("Cannot interact an orbital with another orbital bonded to same atom!")

        if self.neighbor is None and other.neighbor is not None:
            other.unpair(2 - self.num_electrons)
            self.pair(other)

        else:
            raise ValueError("Invalid interaction!")

    def interact_single_electron_transfer(self, other: 'ValenceOrbital') -> None:
        # interacts radical orbital with empty orbital, transferring electrons
        if self.atom is other.atom:
            raise ValueError("Cannot interact an orbital with an orbital on the same atom!")

        if self.neighbor is other:
            raise ValueError("Cannot interact an orbital with its neighbor!")

        if self.num_electrons == 1 and other.num_electrons == 0:
            self.single_electron_transfer(other)

        else:
            raise ValueError("Invalid interaction!")


    def interact(self, other: 'ValenceOrbital') -> None:
        """Interacts two ValenceOrbitals, automatically updating the number of electrons in each
        as well as associated RDKit atoms"""

        # Deprecated; split into above functions

        if self.atom is other.atom:
            raise ValueError("Cannot interact an orbital with an orbital on the same atom!")

        if self.neighbor is other:
            raise ValueError("Cannot interact an orbital with its neighbor!")

        if self.neighbor is None and other.neighbor is None:
            if self.num_electrons + other.num_electrons == 3:
                raise ValueError("Cannot interact filled orbital with radical orbital!")
            elif self.num_electrons + other.num_electrons == 2:
                self.pair(other)
            elif self.num_electrons + other.num_electrons == 1:
                if self.num_electrons == 1:
                    self.single_electron_transfer(other)
                else:
                    other.interact(self)
            else:
                raise ValueError("Cannot interact two empty valence orbitals!")

        elif self.neighbor is None and other.neighbor is not None:
            other.unpair(2 - self.num_electrons)
            self.pair(other)
            logging.info("Interacting unpaired orbital with paired orbital")

        elif self.neighbor is not None and other.neighbor is None:
            other.interact(self)

        elif self.neighbor and other.neighbor:
            raise ValueError("Cannot interact two paired orbitals!")


    def __str__(self) -> str:
        return f"self.num_electrons: {self.num_electrons}; self.atom: {str(self.atom)}; self.neighbor: {str(self.neighbor.atom) if self.neighbor else None}"

    def __repr__(self) -> str:
        return self.__str__()


class Atom:
    def __init__(self, molecule: 'Molecule', atom_type: str, idx: int, num_valence_electrons: int):
        self.molecule = molecule
        self.atom_type = atom_type
        self.idx = idx
        self.valence_orbitals = []
        self.valence_electrons = num_valence_electrons
        num_valence_orbitals = atom_to_num_VOs(self.atom_type)

        # num_valence_electrons // num_valence_orbitals splits up all valence electrons that can be split
        # then have (num_valence_electrons % num_valence_orbitals) left over electrons, which would be assigned directly
        # TODO: fix so that hypervalency is supported. currently, just assume 8 valence electron max for all non-H atoms

        num_divisible_electrons = num_valence_electrons // num_valence_orbitals
        num_leftover_electrons = num_valence_electrons % num_valence_orbitals
        electrons_in_orbitals = [num_divisible_electrons + (1 if idx < num_leftover_electrons else 0) for idx in range(num_valence_orbitals)]

        for idx, num_electrons in enumerate(electrons_in_orbitals):
            self.valence_orbitals.append(ValenceOrbital(num_electrons=num_electrons, atom=self, neighbor=None))

    def make_vo_pairing(self, other_atom: 'Atom', radical_only=False):
        """
        Pairs one VO between this atom and other_atom. The VOs must both be
        currently unpaired and must have total number of electrons = 2.
        If radical_only is true, then only orbitals with one electron each are paired.
        This is used during initialization only.
        """
        for valence_orbital in self.valence_orbitals:
            if valence_orbital.neighbor is None:
                for other_valence_orbital in other_atom.valence_orbitals:
                    if other_valence_orbital.neighbor is None:
                        if valence_orbital.num_electrons + other_valence_orbital.num_electrons == 2:
                            if not radical_only or (valence_orbital.num_electrons == 1 and other_valence_orbital.num_electrons == 1):
                                valence_orbital.pair(other_valence_orbital)
                                return True
    
        raise ValueError("No possible bond between these atoms!")
        return False

    def get_rdkit_atom(self) -> 'rdkit.Chem.rdchem.Atom':
        """Returns the rdkit atom object corresponding to this Atom"""
        return self.molecule.molecule.GetAtomWithIdx(self.idx)

    def get_formal_charge(self) -> int:
        """Returns the Formal Charge on the RDKit Atom"""
        return self.get_rdkit_atom().GetFormalCharge()

    def set_formal_charge(self, charge: int) -> None:
        """Sets the Formal Charge on the RDKit Atom to the charge"""
        self.get_rdkit_atom().SetFormalCharge(charge)

    def get_num_radical_electrons(self) -> int:
        """Returns the number of radical electrons on the RDKit Atom"""
        return self.get_rdkit_atom().GetNumRadicalElectrons()

    def set_num_radical_electrons(self, num_radicals: int) -> None:
        """Sets the number of radical electrons on the RDKit Atom to num_radicals"""
        if num_radicals < 0:
            raise ValueError("err: num_radicals < 0")
        self.get_rdkit_atom().SetNumRadicalElectrons(num_radicals)

    def increment_formal_charge(self) -> None:
        """Increments the Formal Charge on the RDKit Atom by 1"""
        self.set_formal_charge(self.get_formal_charge() + 1)

    def decrement_formal_charge(self) -> None:
        """Decrements the Formal Charge on the RDKit Atom by 1"""
        self.set_formal_charge(self.get_formal_charge() - 1)

    def increment_radical_electrons(self) -> None:
        """Increments the number of radical electrons on the RDKit Atom by 1"""
        self.set_num_radical_electrons(self.get_num_radical_electrons() + 1)

    def decrement_radical_electrons(self) -> None:
        """Decrements the number of radical electrons on the RDKit Atom by 1"""
        self.set_num_radical_electrons(self.get_num_radical_electrons() - 1)

    def increment_bond_order(self, other_atom: 'Atom') -> None:
        """Increments the bond order of the bond on the RDKit Atom and other_atom"""
        rwmol = Chem.RWMol(self.molecule.molecule)
        current_bond = self.molecule.molecule.GetBondBetweenAtoms(self.idx, other_atom.idx)

        if current_bond is None:
            rwmol.AddBond(self.idx, other_atom.idx, Chem.rdchem.BondType.SINGLE)

        elif current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
            raise ValueError("Cannot increment bond order beyond triple!")
            return

        elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
            rwmol.RemoveBond(self.idx, other_atom.idx)
            rwmol.AddBond(self.idx, other_atom.idx, Chem.rdchem.BondType.TRIPLE)

        elif current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
            rwmol.RemoveBond(self.idx, other_atom.idx)
            rwmol.AddBond(self.idx, other_atom.idx, Chem.rdchem.BondType.DOUBLE)

        else:
            logging.critical("Unknown bond type!")
            return

        self.molecule.molecule = rwmol

        Chem.SanitizeMol(self.molecule.molecule, sanitizeOps=sanitizeOps)
        self.molecule.smi = Chem.MolToSmiles(self.molecule.molecule)

    def decrement_bond_order(self, other_atom: 'Atom') -> None:
        """Decrements the bond order of the bond on thes RDKit Atom and other_atom"""
        current_bond = self.molecule.molecule.GetBondBetweenAtoms(self.idx, other_atom.idx)
        if current_bond is None:
            raise ValueError("No bond between atoms!")

        rwmol = Chem.RWMol(self.molecule.molecule)
        if current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
            rwmol.RemoveBond(self.idx, other_atom.idx)
            rwmol.AddBond(self.idx, other_atom.idx, Chem.rdchem.BondType.DOUBLE)

        elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
            rwmol.RemoveBond(self.idx, other_atom.idx)
            rwmol.AddBond(self.idx, other_atom.idx, Chem.rdchem.BondType.SINGLE)

        elif current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
            rwmol.RemoveBond(self.idx, other_atom.idx)

        else:
            logging.critical("Unknown bond type!")
            return

        self.molecule.molecule = rwmol
        Chem.SanitizeMol(self.molecule.molecule, sanitizeOps=sanitizeOps)
        self.molecule.smi = Chem.MolToSmiles(self.molecule.molecule)

    def get_valence_orbitals_to_other_atom(self, atom: 'Atom') -> List['ValenceOrbital']:
        """
        Returns all of self's valence orbitals that are bonded to the specified atom.
        """
        bonded_valence_orbitals = []
        for self_vo in self.valence_orbitals:
            if self_vo.neighbor is not None and self_vo.neighbor.atom is atom:
                bonded_valence_orbitals.append(self_vo)

        return bonded_valence_orbitals

    def get_unpaired_valence_orbitals(self) -> List['ValenceOrbital']:
        return [vo for vo in self.valence_orbitals if vo.neighbor is None]

    def __str__(self) -> str:
        return f"{self.idx}, {self.atom_type}"

    def __repr__(self) -> str:
        return self.__str__()


class Molecule:
    def __init__(self, smi: str):
        self.smi = smi
        self.molecule = Chem.RWMol()
        self.orig_molecule = Chem.MolFromSmiles(smi)
        self.orig_molecule = Chem.AddHs(self.orig_molecule)  # always add H's to make bonding correct
        Chem.Kekulize(self.orig_molecule)  # change to kekulized smiles to remove aromatic bonds
        self.num_atoms = self.orig_molecule.GetNumAtoms()

        # Create adjacency list representation for bonds. Initial_bonds is not symmetric.
        initial_bonds: Dict[int, List[int]] = dict()
        for bond in self.orig_molecule.GetBonds():
            bond.SetIsAromatic(False)  # remove aromaticity properties
            atom_1 = bond.GetBeginAtomIdx()
            atom_2 = bond.GetEndAtomIdx()
            num_bonds = round(bond.GetBondTypeAsDouble())
            initial_bonds[atom_1] = initial_bonds.get(atom_1, []) + [atom_2] * num_bonds

        # Create Atom objects
        self.atoms = []
        rd_periodic_table = Chem.GetPeriodicTable()
        for idx, atom in enumerate(self.orig_molecule.GetAtoms()):
            atom.SetIsAromatic(False)  # remove aromaticity properties
            num_valence_electrons = rd_periodic_table.GetNOuterElecs(atom.GetSymbol()) - atom.GetFormalCharge()
            rdkit_atom = Chem.Atom(atom.GetSymbol())
            rdkit_atom.SetFormalCharge(atom.GetFormalCharge())
            rdkit_atom.SetNumRadicalElectrons(atom_to_num_VOs(atom.GetSymbol()) - abs(num_valence_electrons - atom_to_num_VOs(atom.GetSymbol())))
            rdkit_atom.SetNoImplicit(True) # prevent additional hydrogens from extraneously being added
            self.molecule.AddAtom(rdkit_atom)
            self.atoms.append(Atom(molecule=self, atom_type=atom.GetSymbol(), idx=idx, num_valence_electrons=num_valence_electrons))

        # Create all bonds as part of ValenceOrbital objects
        for atom_idx, neighbors in initial_bonds.items():
            atom = self.atoms[atom_idx]
            for neighbor_atom_idx in neighbors:
                atom.make_vo_pairing(self.atoms[neighbor_atom_idx], radical_only=True)

        Chem.SanitizeMol(self.molecule, sanitizeOps=sanitizeOps)

    def get_atom_by_idx(self, idx: int) -> Optional['Atom']:
        """Returns the Atom object at the specific index of the molecule"""
        if idx >= self.num_atoms or idx < 0:
            raise ValueError("get_atom_by_idx(): idx out of Range")

        return self.atoms[idx]

    def get_vos_between(self, atom1: 'Atom', atom2: 'Atom') -> List['ValenceOrbital']:
        """Returns all atom1-valence orbitals that are bonded to atom2"""
        return atom1.get_valence_orbitals_to_other_atom(atom2)

    def get_vos_between_by_idx(self, atom1_idx: int, atom2_idx: int) -> List['ValenceOrbital']:
        """Returns all atom1-valence orbitals that are bonded to atom2"""
        return self.get_vos_between(self.get_atom_by_idx(atom1_idx), self.get_atom_by_idx(atom2_idx))

    def get_all_valence_orbitals(self) -> List['ValenceOrbital']:
        """Returns all valence orbitals in the molecule"""
        return [vo for atom in self.atoms for vo in atom.valence_orbitals]

    def get_bond_order(self, atom1: 'Atom', atom2: 'Atom') -> int:
        """Returns the bond order between two indices"""
        return len(atom1.get_valence_orbitals_to_other_atom(atom2))

    def get_bond_order_by_idx(self, atom1_idx: int, atom2_idx: int) -> int:
        """Calls get_bond_order() with atom indices rather than atom objects."""
        atom1, atom2 = self.get_atom_by_idx(atom1_idx), self.get_atom_by_idx(atom2_idx)
        return self.get_bond_order(atom1, atom2)

    def get_unpaired_vos(self, atom: 'Atom') -> List['ValenceOrbital']:
        return atom.get_unpaired_valence_orbitals()

    def get_unpaired_vos_by_idx(self, atom_idx: int) -> List['ValenceOrbital']:
        return self.get_unpaired_vos(self.get_atom_by_idx(atom_idx))

    def unpair_vo(self, atom1: 'Atom', atom2: 'Atom', num_electrons_remaining: int) -> None:
        """Unpairs one of the valence orbitals between the two specified atoms"""
        vos = self.get_vos_between(atom1, atom2)
        if len(vos) == 0:
            raise ValueError("unpair_vo(): No valence orbitals to unpair")
        vos[0].unpair(num_electrons_remaining=num_electrons_remaining)

    def unpair_vo_by_idx(self, atom1_idx: int, atom2_idx: int, num_electrons_remaining: int) -> None:
        """Calls unpair_vo() with atom indices rather than atom objects."""
        atom1, atom2 = self.get_atom_by_idx(atom1_idx), self.get_atom_by_idx(atom2_idx)
        return self.unpair_vo(atom1, atom2, num_electrons_remaining=num_electrons_remaining)

    def make_vo_pairing(self, atom1, atom2) -> None:
        """Makes a pairing between two atoms"""
        atom1.make_vo_pairing(atom2)

    def make_vo_pairing_by_idx(self, atom1_idx: int, atom2_idx: int) -> None:
        """Makes a pairing between two atoms"""
        atom1, atom2 = self.get_atom_by_idx(atom1_idx), self.get_atom_by_idx(atom2_idx)
        self.make_vo_pairing(atom1, atom2)

    def get_all_atomic_charges(self) -> List[int]:
        """Returns a list of all atomic charges in the molecule"""
        return [atom.get_formal_charge() for atom in self.atoms]

    def get_all_atomic_radicals(self) -> List[int]:
        """Returns a list of the number of radicals of each atom the molecule"""
        return [atom.get_num_radical_electrons() for atom in self.atoms]

    """
    def to_pytorch_graph(self, vo_features_only=False) -> 'torch_geometric.data.Data':
        #Convert a molecule to a PyTorch Geometric Data object.
        # Get VO to idx correspondences
        all_vos = self.get_all_valence_orbitals()
        vo_to_idx = {vo : idx for idx, vo in enumerate(all_vos)}
        pt = Chem.GetPeriodicTable()

        # Make tensor of VOs, with atomic features
        vo_tensors = []
        for vo in all_vos:
            if vo_features_only:
                n_electron_one_hot = torch.tensor(one_hot_encode(vo.num_electrons, n_dimensions=4, offset=-1), dtype=torch.float)
                bond_order_one_hot = torch.tensor(one_hot_encode(self.get_bond_order(vo.atom, vo.neighbor.atom) if vo.neighbor else 0, n_dimensions=4, offset=0), dtype=torch.float)
                non_one_hot_properties = torch.tensor([vo.num_electrons, self.get_bond_order(vo.atom, vo.neighbor.atom) if vo.neighbor else 0], dtype=torch.float)
                vo_tensor = torch.cat([n_electron_one_hot, bond_order_one_hot, non_one_hot_properties])
            else:
                n_electron_one_hot = torch.tensor(one_hot_encode(vo.num_electrons, n_dimensions=4, offset=-1), dtype=torch.float)
                bond_order_one_hot = torch.tensor(one_hot_encode(self.get_bond_order(vo.atom, vo.neighbor.atom) if vo.neighbor else 0, n_dimensions=4, offset=0), dtype=torch.float)
                atomic_number_one_hot = torch.tensor(one_hot_encode(vo.atom.get_rdkit_atom().GetAtomicNum(), n_dimensions=100, offset=0), dtype=torch.float)
                degree_one_hot = torch.tensor(one_hot_encode(vo.atom.get_rdkit_atom().GetDegree(), n_dimensions=11, offset=0), dtype=torch.float)
                explicit_valence_one_hot = torch.tensor(one_hot_encode(vo.atom.get_rdkit_atom().GetExplicitValence(), n_dimensions=15, offset=0), dtype=torch.float)
                # formal_charge_one_hot = torch.tensor(one_hot_encode(vo.atom.get_rdkit_atom().GetFormalCharge(), n_dimensions=9, offset=-4), dtype=torch.float)
                # radicals_one_hot = torch.tensor(one_hot_encode(vo.atom.get_rdkit_atom().GetNumRadicalElectrons(), n_dimensions=8, offset=0), dtype=torch.float)
                hybridization_one_hot = torch.tensor(one_hot_encode(vo.atom.get_rdkit_atom().GetHybridization(), n_dimensions=8, offset=0), dtype=torch.float)

                non_one_hot_properties = torch.tensor([vo.num_electrons,
                                                      self.get_bond_order(vo.atom, vo.neighbor.atom) if vo.neighbor else 0,
                                                      vo.atom.idx,
                                                      vo.neighbor.atom.idx if vo.neighbor else -1,
                                                      vo.atom.get_rdkit_atom().GetAtomicNum(),
                                                      vo.atom.get_rdkit_atom().GetDegree(),
                                                      vo.atom.get_rdkit_atom().GetExplicitValence(),
                                                      vo.atom.get_rdkit_atom().GetFormalCharge(),
                                                      vo.atom.get_rdkit_atom().GetNumRadicalElectrons()], dtype=torch.float)

                vo_tensor = torch.cat([n_electron_one_hot, 
                                       bond_order_one_hot, 
                                       atomic_number_one_hot, 
                                       degree_one_hot, 
                                       explicit_valence_one_hot, 
                                       # formal_charge_one_hot, 
                                       # radicals_one_hot, 
                                       hybridization_one_hot, 
                                       non_one_hot_properties])
            vo_tensors.append(vo_tensor)
        
        x = torch.stack(vo_tensors) # x is all the node features

        graph_edges = []
        for vo in all_vos:
            for vo_same_atom in vo.atom.valence_orbitals:
                if vo_same_atom is vo: 
                    continue
                graph_edges.append((vo, vo_same_atom, [1, 0, 0])) # different edge representations for different types of graph edges
            if vo.neighbor:
                graph_edges.append((vo, vo.neighbor, [0, 1, 0]))
            graph_edges.append((vo, vo, [0, 0, 1]))

        # Edge Type: [1, 0, 0] = same atom VO connections, [0, 1, 0] = different atom VO connections, [0, 0, 1] = same VO (self loop)
        edge_info = torch.tensor([[vo_to_idx[vo1], vo_to_idx[vo2], *_type] for vo1, vo2, _type in graph_edges], dtype=torch.long)
        edge_index = edge_info[:, :2].t().contiguous()
        edge_attr = edge_info[:, 2:].float()
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    """
    def __str__(self) -> str:
        return self.smi

    def __repr__(self) -> str:
        return self.smi