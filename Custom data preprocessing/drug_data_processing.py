import numpy as np
import pandas as pd
import os
import sklearn.preprocessing as sk

"""
Preprocessing of the DRUG consumption dataset
http://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29#

The script preprocesses the DRUG consumption dataset and saves the .npz files in Fairness_attack/data
"""

print("===============================================")
print("Preprocessing the DRUG Consumption Dataset...\n")

# Load .csv file
path = 'drug/drug_consumption.data'

# Choose 
col_list = ["ID", "age", "gender", "education", "country", "ethnicity", "nscore", "escore", "oscore", "ascore", "cscore", "impulsive", "ss", "coke"]
data = pd.read_csv(path, header=None, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,20])

# Group classes to transform problem in binary classification: "CL0" (Never Used) form class 1 (Non-user)
# All the others ("CL1", "CL2", "CL3", "CL4", "CL5", "CL6") form class 0 (User)
# (although this may be counterintuitive, a study of the dataset provided by the authors seems to reveal  
#  that drug users have label 0 and non-users have label 1)
data[20] = data[20].map({'CL0':1, 'CL1':0, 'CL2':0, 'CL3':0, 'CL4':0, 'CL5':0, 'CL6':0})

# Convert to int
data[20] = data[20].astype(int)

# Shuffle Dataset before splitting in training and test set.
# (Again, this is a step that is not strictly necessary but used by the original author. We follow
#  their procedure to ensure the reproducibility of the results)
shuffled = data.sample(frac=1,random_state=2021).reset_index(drop=True)

# Create advantaged and disadvantaged groups
group_label = shuffled[2].to_numpy()
# Map -0.4826 (Male) to 0
# Map 0.48246 (Female) to 1
group_label = np.where(group_label<0,0,1)

# Split to data points and ground truths
X_unordered = shuffled.iloc[:, :-1].values

# Move the sensitive feature to index 0 so that it is selected by default
sensitive_feature = X_unordered[:,2] # (gender)
X = np.hstack((sensitive_feature[..., np.newaxis], X_unordered[:,:2], X_unordered[:,3:]))
Y = shuffled.iloc[:, -1].values

# Standardize data column-wise
scaler = sk.StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f'Shape of the datapoints:           {X_scaled.shape}')
print(f'Shape of the corresponding labels: {Y.shape}\n')

# Splitting of the data into training and test sets. Although the paper specifies a 80/20 splitting,
# the shape of their dataset matrices (provided as npz) suggests that the authors used a hard-coded 
# index for the splitting (being 1500). Again, for the sake of results reproduction, we adhere to 
# their procedure

#idx = round(0.8*len(X_scaled))
idx = 1500

X_train = X_scaled[:idx]
X_test = X_scaled[idx:]
Y_train = Y[:idx]
Y_test = Y[idx:]
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape:  {X_test.shape}')
print(f'Y_train shape: {Y_train.shape}')
print(f'Y_test shape:  {Y_test.shape}')

# Create output folder if it doesn't exist
if not os.path.exists('../Fairness_attack/data'):
    os.makedirs('../Fairness_attack/data')
# Make a .npz file for the training and test datasets
np.savez_compressed('../Fairness_attack/data/drug_data.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
# Make a .npz file for the group labels
np.savez_compressed('../Fairness_attack/data/drug_group_label.npz', group_label=group_label)

print("===============================================")