import numpy as np
import pandas as pd
import os

# http://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29 

# Load .csv file
path = 'german/german_final.csv'
data = pd.read_csv(path, header=None)
print(data)

# One-hot-encoding of categorical attributes
# https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)

attributes_to_encode = [0,2,3,5,6,9,11,13,14,16,18,19]

for attribute in attributes_to_encode:
    data = encode_and_bind(data, attribute)


# Group classes (i.e. [A91, A93, A94] as male (0), [A92, A95] as female (1))
data[8] = data[8].map({'A91':0, 'A92':1, 'A93':0, 'A94':0, 'A95':1})

# To increase readibility, map a good risk value (1) to 0 and a bad risk value (2) to 1
data[20] = data[20].map({1: 0, 2:1})


print('Mapped Dataset:\n')
print(data)

# Check NaN rows and drop them
data.dropna(inplace=True)

# Shuffle Dataset
#data_shuffled = data.sample(frac=1, random_state=0)

# Create advantaged and disadvantaged groups: if it's a male (1) map to 0, if it's a female (2) map to 1
group_label = data[8].to_numpy()
print(f'group_label shape: {group_label.shape}\n')
print(f'group_label: {group_label}\n')

# Standardize
data_normalized=(data-data.mean())/data.std()
print(f'{data_normalized}\n')
# Save label column
data_normalized[20] = data[20]

# Move label column to last column
label = data_normalized.pop(20)
data_normalized = pd.concat([data_normalized, label], 1)

print(data_normalized)

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
print(f'X_train: {X_train}, shape: {X_train.shape}\n')
print(f'X_test: {X_test}, shape: {X_test.shape}\n')
print(f'Y_train: {Y_train}, shape: {Y_train.shape}\n')
print(f'Y_test: {Y_test}, shape: {Y_test.shape}\n')

# Create output folder if it doesn't exist
if not os.path.exists('processed'):
    os.makedirs('processed')
# Make a .npz file for the training and test datasets
np.savez_compressed('processed/german_data.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
# Make a .npz file for the groups
np.savez_compressed('processed/german_group_label.npz', group_label=group_label)
