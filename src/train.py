import torch
import ham_matrix
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math

def homo_lumo(occupations, device):
    idx_temp = np.nonzero(occupations)[0]
    homo_idx = np.argmax(idx_temp)
    lumo_idx = homo_idx + 1
    homo_lumo = torch.zeros(len(occupations)).to(device)
    homo_lumo[int(homo_idx)] = -1
    homo_lumo[int(lumo_idx)] = 1
    return homo_lumo

def evaluateModel(model, test_smiles, test_gaps, loss_fn, value_calculated, element_list, orbital_list):
    model.eval()
    pred = model(test_smiles, value_calculated, element_list, orbital_list)
    loss = loss_fn(torch.Tensor(test_gaps), pred)
    return pred, loss

class Huckel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(6,7,5,5, requires_grad=True))

    def forward(self, smiles, value_calculated, element_list, orbital_list, device):
        result = torch.zeros(len(smiles)).to(device)
        for index, smi in enumerate(smiles):
            all_indices, ham_size = ham_matrix.grabIndices(smi, element_list, orbital_list)
            ham = torch.zeros(ham_size, ham_size).to(device)
            for index_set in all_indices:
                ham_index = index_set[0]
                par_index = index_set[1]
                adjust = torch.zeros(6,7,5,5).to(device)
                adjust[par_index[0]][par_index[1]][par_index[2]][par_index[3]] = 1
                w1 = torch.mul(self.weights, adjust)
                part_ham = torch.zeros(ham_size, ham_size).to(device)
                part_ham[ham_index[0]][ham_index[1]] = 1
                for non in w1.nonzero():
                    w2 = torch.mul(w1[non[0]][non[1]][non[2]][non[3]], part_ham).to(device)
                ham = torch.add(ham, w2)
            w3 = torch.linalg.eigvalsh(ham)
            occupations = ham_matrix.grabOccupations(smi, w3.cpu().detach().numpy(), len(w3))
            w4 = eval(value_calculated + "(occupations, device)")
            final_val = torch.dot(w3, w4)
            result[index] = final_val
        return result

def train_model(train_smiles, train_gaps, test_smiles, test_gaps, folder_name, epochs, lr, bs, loss_func, optimizer, value_calculated, element_list, orbital_list):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Create DataLoader
    train_smiles_index = range(len(train_smiles))
    train_smiles_tensor = torch.Tensor(train_smiles_index).to(device)
    train_gaps_tensor = torch.Tensor(train_gaps).to(device)
    train_data_tensor = TensorDataset(train_smiles_tensor, train_gaps_tensor)
    train_loader = DataLoader(dataset=train_data_tensor, batch_size=bs, shuffle=True)

    # Model Parameters
    model = Huckel().to(device)
    optimizer = eval(optimizer + "(model.parameters(), lr=lr)")
    loss_fn = eval(loss_func)

    # Values to Save
    test_loss = []
    train_loss = []
    min_loss_eval_preds = []

    # Training Loop (with evaluations at end of every epoch)
    min_loss = math.inf
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            model.train()
            optimizer.zero_grad()
            model_input = []
            for index, val in enumerate(x_batch):
                model_input.append(train_smiles[int(val)])
            pred = model(model_input, value_calculated, element_list, orbital_list, device)
            loss = loss_fn(y_batch, pred)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
        print("Epoch " + str(epoch) + " complete", flush=True)
        train_loss.append(epoch_loss/epochs)
        eval_preds, loss_again = evaluateModel(model, test_smiles, test_gaps, loss_fn, value_calculated, element_list, orbital_list)
        test_loss.append(float(loss_again))
        print("Train Loss: " + str(epoch_loss/epochs), flush=True)
        print("Test Loss: " + str(float(loss_again)), flush=True)
        if float(loss_again) < min_loss:
            min_loss = float(loss_again)
            min_loss_eval_preds = eval_preds
            torch.save(model, folder_name + "/min_loss.pt")

    return train_loss, test_loss, min_loss_eval_preds
