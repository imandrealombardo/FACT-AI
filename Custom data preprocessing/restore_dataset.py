import numpy as np
import pandas as pd

path = 'drug/drug_consumption.data'
col_list = ["ID", "age", "gender", "education", "country", "ethnicity", "nscore", "escore", "oscore", "ascore", "cscore", "impulsive", "ss", "coke"]
data = pd.read_csv(path, header=None, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,20])
# Group classes to transform problem in binary classification: "CL0" (Never Used) and "CL1" (Used over a Decade Ago) form class 0 (Non-user)
# All the others ("CL2", "CL3", "CL4", "CL5", "CL6") form class 1 (User)
data[20] = data[20].map({'CL0':1, 'CL1':0, 'CL2':0, 'CL3':0, 'CL4':0, 'CL5':0, 'CL6':0})
# Convert to int
data[20] = data[20].astype(int)
# Split to data points and ground truths
X_unordered = data.iloc[:, :-1].values
sex = X_unordered[:,2]
X = np.hstack((X_unordered[:,:2], X_unordered[:,3:], sex[..., np.newaxis]))


a = np.load('/Users/matteo/Downloads/data_author/drug_data.npz')
print('Drug data - authors')
print(a.files)
authors_X_train = a['X_train']
authors_X_test = a['X_test']
authors_Y_train = a['Y_train']
authors_Y_test = a['Y_test']

# b = np.load('/Users/matteo/Dropbox/Università/Module 3/FACT/Project/FACT-AI/Custom data preprocessing/processed/drug_data.npz')
# print('Drug data - custom')
# print(b.files)
# our_X_train = b['X_train']
# our_X_test = b['X_test']
# our_Y_train = b['Y_train']
# our_Y_test = b['Y_test']

transformed_data = np.zeros_like(authors_X_train)

for col in range(authors_X_train.shape[1]):
    mean = np.mean(X[:,col])
    std = np.std(X[:,col])

    print(f"Col {col}: mean {mean} | std: {std}")

    transformed_data[:,col] = authors_X_train[:,col] * std + mean
    if col == 0:
        transformed_data[:,col] = np.sort(transformed_data[:,col])

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
with open('restored_drug.txt', 'w') as f:
    f.write(np.array2string(transformed_data, separator=', '))