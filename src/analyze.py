import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def analyze_results(train_loss, test_loss, min_loss_eval_preds, test_gaps, folder_name, loss_curve, lowest_loss_data, lowest_loss_graph):
    if loss_curve == True:
        epochs = len(train_loss)
        plt.plot(range(epochs), train_loss, 'r-', range(epochs), test_loss, 'b-')
        plt.legend(['Training Loss', 'Evaluation Loss'])
        plt.xlabel("Epochs")
        plt.ylabel("Loss (eV)")
        plt.savefig(folder_name + "/loss_curve.png")
        plt.clf()
    if lowest_loss_data == True:
        all_data = {"Actual": test_gaps, "Predicted": min_loss_eval_preds.detach().numpy()}
        df = pd.DataFrame(all_data)
        df.to_csv(folder_name + "/prediction_data.csv")
    if lowest_loss_graph == True:
        plt.plot(test_gaps, min_loss_eval_preds.detach().numpy(), 'ro')
        plt.xlabel("Actual HOMO LUMO Gaps (eV)")
        plt.ylabel("Predicted HOMO LUMO Gaps (eV)")
        plt.savefig(folder_name + "/prediction_graph.png")
