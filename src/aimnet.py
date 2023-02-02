import numpy as np
import torch  # type: ignore
import os
import sys
import logging
from rdkit.Chem import Descriptors
from torch import Tensor
from typing import Dict, Optional
from rdkit import Chem  # type: ignore


class AimnetCalculator():
    """
    Uses the AIMNet-NSE model by Zubatyuk et al to calculate molecular energies with Machine Learning.
    Reference: https://chemrxiv.org/engage/chemrxiv/article-details/60c75793702a9b15d318cb0c
    Parts of this code is from the model's associated Github Repo, and the models used were downloaded from the associated Zenodo link.
    """

    def __init__(self, models_base_path: Optional[str] = None):
        if models_base_path is None:
            models_base_path = os.path.join(os.path.dirname(__file__), 'aimnet_models')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rd_periodic_table = Chem.GetPeriodicTable()
        self.models = [torch.jit.load(os.path.join(models_base_path, 'aimnet-nse-cv{}.jpt'.format(i))).to(self.device) for i in range(5)]

    def predict(self, data: Dict[str, Tensor], smi: str) -> float:
        """
        Predicts the energy of a molecule in tensorial representation
        Returns DFT calculations for single atom (with SMILES specified in smi)
        """
        from utils import get_orca_energy
        if data['numbers'].size(dim=1) == 1:
            # if the molecule is a single atom, use Orca/DFT to calculate the energy since AIMNet doesn't work
            if smi == '[H+]':
                return 0.0 # proton has 0 energy
            return get_orca_energy(smi, geometry_optimizer='rdkit')
        # change data to have alpha and beta total charge
        data['charge'] = 0.5 * torch.stack([data['charge'] - data['mult'] + 1, data['charge'] + data['mult'] - 1], dim=-1)
        return float(np.mean([model(data)['energy'][-1].cpu().numpy() for model in self.models]).tolist())

    def mol2data(self, smi: str, charge: float, mult: float, optimizer: str = 'xtb') -> Dict[str, Tensor]:
        """
        Changes a smiles string to the coordinate tensorial representation needed by the model.
        If optimizer == 'rdkit', then the default rdkit geometry optimization force field is used.
        If optimizer == 'xtb', then the BFGS method from xtb is used.
        """
        from utils import smi_to_coords
        atoms, coords = smi_to_coords(smi, optimizer=optimizer)
        coord = np.array(coords)
        numbers = np.array([self.rd_periodic_table.GetAtomicNumber(a) for a in atoms])
        coord = torch.tensor(coord, dtype=torch.float).unsqueeze(0).repeat(1, 1, 1).to(self.device)
        numbers = torch.tensor(numbers, dtype=torch.long).unsqueeze(0).repeat(1, 1).to(self.device)
        charge = torch.tensor([charge]).to(self.device)  # cation, neutral, anion
        mult = torch.tensor([mult]).to(self.device)
        return dict(coord=coord, numbers=numbers, charge=charge, mult=mult)

    def get_energy(self, smi: str, optimizer: str = 'xtb') -> float:
        """
        Returns the energy predicted by AIMNet-NSE on the given molecular system, in eV.
        If optimizer == 'rdkit', then the default rdkit geometry optimization force field is used.
        If optimizer == 'xtb', then the BFGS method from xtb is used.
        """

        molecules = smi.split('.')
        system_energy = 0.0
        for molecule in molecules:
            molecule_charge = Chem.rdmolops.GetFormalCharge(Chem.MolFromSmiles(molecule))
            num_unpaired_electrons = Descriptors.NumRadicalElectrons(Chem.MolFromSmiles(molecule))
            logging.info('Calculating AIMNet energy for {}'.format(molecule))
            spin_multiplicity = num_unpaired_electrons + 1
            data = self.mol2data(molecule, charge=molecule_charge, mult=spin_multiplicity, optimizer=optimizer)
            with torch.jit.optimized_execution(False), torch.no_grad():
                pred = self.predict(data, smi=molecule)
            system_energy += pred
        return system_energy