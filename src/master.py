import yaml
import initialize
import train
import analyze
import os
import time
import sys

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print("Job started at " + str(current_time), flush=True)
start = time.time()

# Load variables from config

experiment = "../config/" + str(sys.argv[1])
stream = open(experiment, 'r')
file_input = yaml.load(stream, Loader=yaml.Loader)

# Make results folder
folder_name = "../results/" + file_input['results']['results_directory']
if os.path.isdir(folder_name) == False:
    os.mkdir(folder_name)

# Load data

train_smiles, train_gaps, test_smiles, test_gaps = initialize.load_data(file_input['data']['train_data'],
                                                                        file_input['data']['test_data'],
                                                                        file_input['data']['input_size'],
									file_input['data']['output_size'])
print("Data Loaded! Moving on to training...", flush=True)

# Train model

train_loss, test_loss, min_loss_eval_preds = train.train_model(train_smiles, train_gaps,
                                                                test_smiles, test_gaps,
                                                                folder_name,
                                                                file_input['model']['epochs'],
                                                                file_input['model']['learning_rate'],
                                                                file_input['model']['batch_size'],
                                                                file_input['model']['loss_fn'],
                                                                file_input['model']['optimizer'],
                                                                file_input['huckel']['value_calculated'],
                                                                file_input['huckel']['elements_considered'],
                                                                file_input['huckel']['orbitals_considered'])
print("Training complete! Moving on to analysis...", flush=True)

# Analyze and save data

analyze.analyze_results(train_loss, test_loss, 
                        min_loss_eval_preds, test_gaps, 
                        folder_name, 
                        file_input['results']['loss_curve'],
                        file_input['results']['lowest_loss_data'],
                        file_input['results']['lowest_loss_graph'])

t2 = time.localtime()
current_time2 = time.strftime("%H:%M:%S", t)
print("Analysis complete! Job complete at " + str(current_time2), flush=True)
end = time.time()
print("This job took " + str(round((end - start)/60, 2)) + " minutes", flush=True)
