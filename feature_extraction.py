


__author__ = 'Stefano Mauceri'
__email__ = 'mauceri.stefano@gmail.com'


"""
In order to run the algorithm you need simply to:
    1)
    - Put your data in the "data" folder using the
    exact same format as the examples provided.
    - Please follow the same method to name
    files.
    - Data is organised as .npy files where we have:
    (n samples, time series length).
    - Data must contain at least two classes.

    2)
    - Go to the "MAIN" section below.
    - Put the name of your dataset in the dataset list.
    - Set the number of features to be extrated.
    - Finally, run the present script.

    3) Collect results in the "results" folder.


If you want to modify the algorithm it is a
completely different story.
Some changes may be easy, others may require
you to modify multiple files across the package.

I may be able to support you if you contact me via email.

For now, I can point you to some of the main files
you may want to work with:
- GRAMMAR: /grammars/tsc_grammar.bnf
- EVOLUTIONARY PARAMETERS: /parameters/tsc_parameters.txt
- FITNESS FUNCTION: /src/fitness/tsc_fitness.py
- FUNCTIONS USED IN THE GRAMMAR (PRIMITIVES): /src/fitness/math_functions.py

"""



# =============================================================================
# IMPORT
# =============================================================================



import os
import re
import subprocess
import numpy as np
import pandas as pd
from src.fitness.math_functions import *



# =============================================================================
# FUNCTIONS
# =============================================================================



def extract_features(X, phenotype):
    T = X
    return eval(phenotype)



def customise_default_grammar(dataset, cwd):
    path_default = os.path.join(cwd, 'grammars', 'tsc_grammar.bnf')
    path_customised = os.path.join(cwd, 'grammars', f'tsc_grammar_{dataset}.bnf')

    grammar = open(path_default, 'r')
    new_grammar = open(path_customised, 'w')

    for line in grammar:
        if line.startswith('<lb>'):
            line = f'<lb> ::= GE_RANGE:{ts_length}\n'
        new_grammar.write(line)

    grammar.close()
    new_grammar.close()



def read_parameters(path):
    df = pd.read_csv(path, sep=':', names=['K', 'V'])
    K = [re.sub(r'\s+', '', i) for i in df.K.values]
    V = [re.sub(r'\s+', '', i) for i in df.V.values]
    return dict(zip(K, V))



def save_last_feature(parameters):

    run = parameters['RUN_NUMBER']
    generations = parameters['GENERATIONS']
    fts = parameters['FEATURE_NUMBER']
    class_ = parameters['CLASS']
    dataset_name = parameters['DATASET_NAME']
    data_dir = os.path.abspath(os.path.join(os.getcwd(), 'data'))

    current_save_dir = os.path.abspath(os.path.join(os.getcwd(), 'results', parameters['EXPERIMENT_NAME']))
    current_save_dir = current_save_dir.split(os.path.sep)[:-2]
    current_save_dir = os.path.sep.join(current_save_dir)

    path = os.path.join(data_dir, dataset_name)
    X = np.load(os.path.join(path, f'{dataset_name}_X_TRAIN.npy'))
    Y = np.load(os.path.join(path, f'{dataset_name}_Y_TRAIN.npy'))
    X_class_c = X[(Y == class_)]

    phen_path = os.path.join(current_save_dir,
                             f'F_{fts}',
                             f'CLASS={class_}_RUN={run}',
                             f'{generations}.txt')

    fts_path = os.path.join(current_save_dir,
                            'FEATURES',
                            f'CLASS={class_}_RUN={run}_F_{fts}.npy')

    phenotype_file = open(phen_path, 'r')
    for line in phenotype_file:
        if line.startswith('Phenotype:'):
            phenotype = next(phenotype_file).strip()
            x = extract_features(X_class_c, phenotype)
            np.save(fts_path, x)
    phenotype_file.close()

    save_phen_path = os.path.join(current_save_dir, 'PHENOTYPES', f'CLASS={class_}_RUN={run}.txt')
    save_phen_file = open(save_phen_path, 'a')
    save_phen_file.write(phenotype + '\n')
    save_phen_file.close()



def write_parameters(parameters, path):
    with open(path, 'w') as new_parameters:
        for k,v in parameters.items():
            new_parameters.write(f'{k}: {v}\n')
    new_parameters.close()



def wrapper(dataset, class_, feature_number, cwd, parameters_file='tsc_parameters.txt', save_last=False):

    path_to_ponyge = os.path.join(cwd, 'src', 'ponyge.py')

    path_default = os.path.join(cwd, 'parameters', f'{parameters_file}')
    parameters_default = read_parameters(path_default)

    parameters = {'CLASS':class_,
                  'FEATURE_NUMBER':feature_number,
                  'DATASET_NAME':dataset,
                  'GRAMMAR_FILE':f'tsc_grammar_{dataset}.bnf',
                  'MUTATE_DUPLICATES':True}

    RUNS = int(parameters_default['RUNS'])
    for run in range(1, RUNS+1):
        exp_name = os.path.sep.join([f'{dataset}_KNN', f'F_{feature_number}', f'CLASS={class_}_RUN={run}'])

        parameters['RUN_NUMBER'] = run
        parameters['EXPERIMENT_NAME'] = exp_name

        parameters_default.update(parameters)
        path_customised_parameters = os.path.join(cwd, 'parameters', f'tsc_parameters_{dataset}_{class_}_{feature_number}.txt')
        write_parameters(parameters_default, path_customised_parameters)

        ex = f'python {path_to_ponyge} --parameters tsc_parameters_{dataset}_{class_}_{feature_number}.txt'
        subprocess.run(ex, shell=True, cwd=os.path.join(cwd, 'src'))

        os.remove(path_customised_parameters)

        if save_last:
            save_last_feature(parameters_default)



# =============================================================================
# MAIN
# =============================================================================


# Target data-sets.
dataset_list = ['SyntheticControl', 'GunPoint']

# Number of features to be extracted.
features = 3


for dataset in dataset_list:


    cwd = os.getcwd()
    path_to_dataset = os.path.join(cwd, 'data', dataset, '')


    # We need to know the time series length.
    # I read this argument from data.
    X = np.load(path_to_dataset + f'{dataset}_X_TRAIN.npy')
    ts_length = X.shape[1]


    # We need to know the positive class label.
    # In this example I run through all
    # the available class labels.
    # I read this argument from data.
    Y = np.load(path_to_dataset + f'{dataset}_Y_TRAIN.npy')
    classes = np.unique(Y)


    # We need to modify the default grammar
    # to account for the current time series length.
    customise_default_grammar(dataset, cwd)


    # Now for each feature and
    # for each class
    # we call the algorithm.
    for f in range(1, features+1):
        for c in classes:
            if f == features:
                save_last = True
            else:
                save_last = False
            wrapper(dataset, c, f, cwd,
                    parameters_file='tsc_parameters.txt',
                    save_last=save_last)



# =============================================================================
# END
# =============================================================================


