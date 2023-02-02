from rdkit.Chem import AllChem  # type: ignore
from rdkit.Chem import Descriptors
from rdkit import Chem  # type: ignore
from suppressor import Suppressor  # type: ignore
import os
from functools import lru_cache
from typing import List, Tuple
import numpy as np
import logging
import subprocess

def one_hot_encode(index: int, n_dimensions: int, offset: int = 0) -> np.ndarray:
    '''
    Given an input index, returns a one-hot encoded array of length n_dimensions
    Subtracts offset from index, so use offset = min(values) to encode negative values.
    '''
    if index < offset:
        raise ValueError("Cannot one-hot encode this value, as it is smaller than offset!")
    if index - offset >= n_dimensions:
        raise ValueError("Cannot one-hot encode this value, as it is greater than n_dimensions!")
    one_hot = np.zeros(n_dimensions)
    one_hot[index - offset] = 1
    return one_hot

def atom_to_num_VOs(atom_symbol: str):
    atom_to_VO_map = {'H': 1, 'S': 6, 'P': 5}
    return atom_to_VO_map.get(atom_symbol, 4)

def output_3d_coords(atoms: List[str], atom_coords: 'np.ndarray', output_format: str = 'xyz') -> str:
    """
    Returns the coordinates of a molecule in the specified output format.
    Supported Output Formats: 'xyz', 'turbo'
    """
    if output_format not in {'xyz', 'turbo'}:
        raise ValueError('Invalid output format. Supported formats: xyz, turbo')

    logging.info('Outputting coordinates in {} format'.format(output_format))
    if output_format == 'xyz':
        output = str(len(atoms)) + '\n\n'
        for atom, (x, y, z) in zip(atoms, atom_coords):
            output += ' '.join([atom, str(x), str(y), str(z)]) + '\n'

    elif output_format == 'turbo':
        output = '$coord'
        for atom, (x, y, z) in zip(atoms, atom_coords):
            output += '\n' + ' '.join([str(x), str(y), str(z), atom.lower()])
        output += '\n$end\n'

    return output


def smi_to_coords(smi: str, optimizer: str = 'rdkit', suppress_output: bool = False) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    """
    Returns the atoms and coordinates of a molecule (given as a smiles string) as arrays.
    Supported Geometry Optimizers: 'rdkit' (ETKDG method), 'xtb' (GFN2-xTB method)
    suppress_output suppresses output from the BFGS geometry optimization when using xtb. Default is True
    """
    if optimizer not in {'rdkit', 'xtb'}:
        raise ValueError('Invalid optimizer. Supported optimizers: rdkit, xtb')

    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)  # for reproducibility
    lines = Chem.MolToMolBlock(mol).split('\n')
    atoms = []
    atom_coords = []

    logging.info('Converting SMILES {} to coordinates'.format(smi))
    # string parsing RDKit mol block
    for idx, line in enumerate(lines):
        if idx < 3:
            continue
        elif idx == 3:
            num_atoms = mol.GetNumAtoms()
        elif idx <= 3 + num_atoms:
            x, y, z, atom = [c for c in line.split(' ') if len(c) > 0][:4]
            x, y, z = map(float, [x, y, z])
            atom_coords.append((x, y, z))
            atoms.append(atom)

    # geometry optimization beyond RDKit with XTB if specified
    if optimizer == 'xtb':
        logging.info('Optimizing geometry with xTB')
        charge = Chem.rdmolops.GetFormalCharge(mol)
        num_unpaired_electrons = Descriptors.NumRadicalElectrons(mol)
        xyzfile = output_3d_coords(atoms, atom_coords, output_format='xyz')
        with open("tmp.xyz", "w") as f:
            f.write(xyzfile)

        command = "xtb tmp.xyz --opt normal --gfn 2 --chrg {} --uhf {}".format(charge, num_unpaired_electrons)
        subprocess.check_call(command.split(), stdout=open('xtblog.txt', 'w'), stderr=open(os.devnull, 'w'))
        
        with open("xtbopt.xyz", "r") as f:
            lines = f.readlines()
            atoms = []
            atom_coords = []
            for line in lines[2:]:
                atom, x, y, z = line.split()
                x, y, z = map(float, [x, y, z])
                atom_coords.append((x, y, z))
                atoms.append(atom)

        for output_file in ['tmp.xyz', 'xtbopt.xyz', 'xtbopt.log', 'xtbtopo.mol', 'xtbrestart', 'wbo', 'tmp.ges', 'charges', 'xtblog.txt']:
            if os.path.exists(output_file):
                os.remove(output_file)

    return atoms, atom_coords


def get_system_energy(smi: str, optimizer: str = 'rdkit', n_attempts: int = 0, n_max_attempts: int = 10) -> float:
    """
    Returns the (potential) energy of the system (given as a smiles string) in eV.
    Uses xTB (GFN2-xTB) calculations for energy calculation. 
    Geometry optimization is done either by RDKit (ETKDG method) or xTB (GFN2-xTB method).
    Tries for up to n_max_attempts, before giving up and throwing an error.
    """
    try:
        molecules = smi.split('.')
        total_energy = 0

        logging.info('Attempt No. {} for getting energy' .format(n_attempts))
        for molecule in molecules:
            logging.info('Getting energy with xTB')
            mol = Chem.MolFromSmiles(molecule)
            atoms, atom_coords = smi_to_coords(molecule, optimizer=optimizer)
            charge = Chem.rdmolops.GetFormalCharge(mol)
            num_unpaired_electrons = Descriptors.NumRadicalElectrons(mol)
            xyzfile = output_3d_coords(atoms, atom_coords, output_format='xyz')
            with open("tmp.xyz", "w") as f:
                f.write(xyzfile)

            command = "xtb tmp.xyz".format(charge, num_unpaired_electrons)
            subprocess.check_call(command.split(), stdout=open('xtblog.txt', 'w'), stderr=open(os.devnull, 'w'))

            with open("xtblog.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if 'TOTAL ENERGY' in line:
                        energy = float(line.split()[-3])
        
            for output_file in ['tmp.xyz', 'xtbopt.xyz', 'xtbopt.log', 'xtbtopo.mol', 'xtbrestart', 'wbo', 'tmp.ges', 'charges', 'xtblog.txt']:
                if os.path.exists(output_file):
                    os.remove(output_file)

            logging.info('Energy of {} is {} hartree'.format(molecule, energy))
            total_energy += energy * 27.2114

    except Exception as e:
        if n_attempts >= n_max_attempts:
            raise e
        else:
            return get_system_energy(smi, optimizer, n_attempts=n_attempts + 1)

    return total_energy

def get_aimnet_energy(smi: str, optimizer: str = 'xtb') -> float:
    from aimnet import AimnetCalculator
    ac = AimnetCalculator()
    return ac.get_energy(smi, optimizer)

def get_orca_energy(smi: str, geometry_optimizer='dft') -> float:
    """
    Returns the (potential) energy of the system (given as a smiles string) in eV.
    Uses ORCA calculations for energy calculation. For geometry optimization, 
    either RDKit (ETKDG method), xTB (GFN2-xTB method), or DFT can be used.
    Use arguments 'rdkit', 'xtb', and 'dft' to call such optimization routines.
    """
    molecules = smi.split('.')
    total_energy = 0

    for molecule in molecules:
        mol = Chem.MolFromSmiles(molecule)
        charge = Chem.rdmolops.GetFormalCharge(mol)
        num_unpaired_electrons = Descriptors.NumRadicalElectrons(mol)
        atoms, atom_coords = smi_to_coords(molecule, optimizer='xtb' if geometry_optimizer == 'xtb' else 'rdkit')
        xyzfile = output_3d_coords(atoms, atom_coords, output_format='xyz')
        with open("tmp.xyz", "w") as f:
            f.write('! PBE0 ma-def2-SVP {}\n'.format('opt' if geometry_optimizer == 'dft' else ''))
            f.write('* xyz {} {}\n'.format(charge, num_unpaired_electrons + 1))
            for idx, line in enumerate(xyzfile.split('\n')):
                if idx > 1:
                    f.write(line + '\n')
            f.write('*')

        command = "orca tmp.xyz"
        subprocess.check_call(command.split(), stdout=open('orca.log', 'w'), stderr=open(os.devnull, 'w'))
        
        with open("orca.log", "r") as f:
            for line in f.readlines():
                if "FINAL SINGLE POINT ENERGY" in line:
                    energy = float(line.split()[4])

        for output_file in ['orca.log', 'tmp.xyz', 'tmp_trj.xyz', 'tmp_property.txt', 'tmp.engrad', 'tmp.opt', 'tmp.gbw', 'tmp.ges', 'tmp.densities']:
            if os.path.exists(output_file):
                os.remove(output_file)

        logging.info('Energy of {} is {} hartree'.format(molecule, energy))
        total_energy += energy * 27.2114

    return total_energy
