import numpy as np
import pandas as pd
import os

"""
Preprocessing of the COMPAS Dataset
https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv

The script preprocesses the COMPAS Dataset and saves the .npz files in Fairness_attack/data
"""

print("===============================================")
print("Preprocessing the COMPAS Dataset...\n")

# Load .csv file
path = 'compas/compas-scores-two-years.csv'

# Choose the features as prediction features. We chose the ones provided in the paper (Table 1)
col_list = ["sex", "juv_fel_count", "priors_count", "race", "age_cat", "juv_misd_count", "c_charge_degree", "juv_other_count", "two_year_recid"]

data = pd.read_csv(path, header=0, usecols=col_list)

# Map categorical/qualitative attributes to numerical ones (Label Encoding)
data["sex"] = data["sex"].map({'Male':0, 'Female':1})
data["race"] = data["race"].map({'African-American':0, 'Asian':1, 'Caucasian':2, 'Hispanic':3, 'Native American':4, 'Other':5})
data["age_cat"] = data["age_cat"].map({'Less than 25':0, '25 - 45':1, 'Greater than 45':2})
data["c_charge_degree"] = data["c_charge_degree"].map({'M':0, 'F':1})

# Create advantaged and disadvantaged groups
group_label = data["sex"].to_numpy()

# (Here, differently from the other datasets, there's no need to move the sensitive features 
#  as it is already positioned at index 0)

# Standardize
data_normalized=(data-data.mean())/data.std()

# Save label column
data_normalized["two_year_recid"] = data["two_year_recid"]

# Split to data points and ground truths
X = data_normalized.iloc[:, :-1].values
Y = data_normalized.iloc[:, -1].values

print(f'\n{X.shape}\n')
print(f'\n{Y.shape}\n')

# Split 80/20
idx = round(0.8*len(X))
X_train = X[:idx]
X_test = X[idx:]
Y_train = Y[:idx]
Y_test = Y[idx:]
print(f'X_train shape: {X_train.shape}\n')
print(f'X_test shape:  {X_test.shape}\n')
print(f'Y_train shape: {Y_train.shape}\n')
print(f'Y_test shape:  {Y_test.shape}\n')

# Create output folder if it doesn't exist
if not os.path.exists('../Fairness_attack/data'):
    os.makedirs('../Fairness_attack/data')
# Make a .npz file for the training and test datasets
np.savez_compressed('../Fairness_attack/data/compas_data.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
# Make a .npz file for the groups
np.savez_compressed('../Fairness_attack/data/compas_group_label.npz', group_label=group_label)

print("===============================================")