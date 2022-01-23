import numpy as np

a = np.load('/Users/matteo/Dropbox/Università/Module 3/FACT/Project/FACT-AI/Fairness_attack/data/german_data.npz')
print('German data list and shapes')
print(a.files)
authors_group_label = a['X_test']


b = np.load('/Users/matteo/Dropbox/Università/Module 3/FACT/Project/FACT-AI/Custom data preprocessing/processed/german_data.npz')
print('German data list and shapes')
print(b.files)
our_group_label = b['X_test']

print(np.all(authors_group_label == our_group_label))

