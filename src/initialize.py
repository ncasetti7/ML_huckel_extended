import numpy as np
import random

def load_data(train_file, test_file, input_size, output_size):
    test_data = np.load("../data/" + test_file, allow_pickle=True)
    train_data = np.load("../data/" + train_file, allow_pickle=True)

    # Parse training and testing data
    test_gaps = []
    test_smiles = []
    train_gaps = []
    train_smiles = []
    for atom in test_data:
        test_gaps.append(atom['homo_lumo_grap_ref'])
        test_smiles.append(atom['smiles'])
    for atom in train_data:
        train_gaps.append(atom['homo_lumo_grap_ref'])
        train_smiles.append(atom['smiles'])
    pairs = list(zip(train_smiles, train_gaps))
    pairs = random.sample(pairs, input_size)
    train_smiles_final, train_gaps_final = zip(*pairs)
    test_smiles = test_smiles[0:output_size]
    test_gaps = test_gaps[0:output_size]
    return train_smiles_final, train_gaps_final, test_smiles, test_gaps
