import numpy as np
import pandas as pd
import os

# http://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29#

# Load .csv file
path = 'drug/drug_consumption.data'
col_list = ["ID", "age", "gender", "education", "country", "ethnicity", "nscore", "escore", "oscore", "ascore", "cscore", "impulsive", "ss", "coke"]
data = pd.read_csv(path, header=None, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,20])
print(data)
print(f'Unique "ID" values: {data[0].unique()}\n')
print(f'Unique "age" values: {data[1].unique()}\n')
print(f'Unique "gender" values: {data[2].unique()}\n')
print(f'Unique "education" values: {data[3].unique()}\n')
print(f'Unique "country" values: {data[4].unique()}\n')
print(f'Unique "ethnicity" values: {data[5].unique()}\n')
print(f'Unique "nscore" values: {data[6].unique()}\n')
print(f'Unique "escore" values: {data[7].unique()}\n')
print(f'Unique "oscore" values: {data[8].unique()}\n')
print(f'Unique "ascore" values: {data[9].unique()}\n')
print(f'Unique "cscore" values: {data[10].unique()}\n')
print(f'Unique "impulsive" values: {data[11].unique()}\n')
print(f'Unique "ss" values: {data[12].unique()}\n')
print(f'Unique "coke" values: {data[20].unique()}\n')

# Group classes to transform problem in binary classification: "CL0" (Never Used) and "CL1" (Used over a Decade Ago) form class 0 (Non-user)
# All the others ("CL2", "CL3", "CL4", "CL5", "CL6") form class 1 (User)
data[20] = data[20].map({'CL0':0, 'CL1':0, 'CL2':1, 'CL3':1, 'CL4':1, 'CL5':1})

# Handle NaN values in data[20] removing rows
# It should remove 19 rows
data.dropna(subset = [20], inplace=True)
#data[20] = data[20].fillna(data[20].median())

# Convert to int
data[20] = data[20].astype(int)

print('Mapped Dataset:\n')
print(data)

# Shuffle Dataset
#data_shuffled = data.sample(frac=1, random_state=0)

# Create advantaged and disadvantaged groups
group_label = data[2].to_numpy()
print('Unique Group label values')
print(np.unique(group_label))
# Map -0.4826 (Male) to 0
# Map 0.48246 (Female) to 1
group_label = np.where(group_label<0,0,1)

print('Group label\n')
print(np.unique(group_label))

# Split to data points and ground truths
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values
#unique = set(Y)
#print(list(unique))

print(f'\n{X.shape}\n')
print(f'\n{Y.shape}\n')

# Split 80/20
idx = round(0.8*len(X))
X_train = X[:idx]
X_test = X[idx:]
Y_train = Y[:idx]
Y_test = Y[idx:]
print(f'X_train: {X_train}, shape: {X_train.shape}\n')
print(f'X_test: {X_test}, shape: {X_test.shape}\n')
print(f'Y_train: {Y_train}, shape: {Y_train.shape}\n')
print(f'Y_test: {Y_test}, shape: {Y_test.shape}\n')

# Create output folder if it doesn't exist
if not os.path.exists('processed'):
    os.makedirs('processed')
# Make a .npz file for the training and test datasets
np.savez_compressed('processed/drug_data.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
# Make a .npz file for the groups
np.savez_compressed('processed/drug_group_label.npz', group_label=group_label)