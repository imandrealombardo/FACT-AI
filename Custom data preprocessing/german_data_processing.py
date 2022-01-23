import numpy as np
import pandas as pd
import os
import sklearn.preprocessing as sk

# http://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29 

# Load .csv file
path = 'german/german_final.csv'
data = pd.read_csv(path, header=None)
print(data)

attributes_to_encode = [0,2,3,5,6,9,11,13,14,16,18,19]
data = pd.get_dummies(data, columns=attributes_to_encode)


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

# Move label column to last column
label = data.pop(20)
data = pd.concat([data, label], 1)

print(data)

# Split to data points and ground truths
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

scaler = sk.StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f'\n{X_scaled.shape}\n')
print(f'\n{Y.shape}\n')

# Split 80/20
idx = round(0.8*len(X_scaled))
X_train = X_scaled[:idx]
X_test = X_scaled[idx:]
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
np.savez_compressed('../Fairness_attack/data/german_data.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

# Make a .npz file for the groups
np.savez_compressed('processed/german_group_label.npz', group_label=group_label)
np.savez_compressed('../Fairness_attack/german_group_label.npz', group_label=group_label)
