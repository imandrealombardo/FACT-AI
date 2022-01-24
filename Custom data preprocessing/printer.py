import numpy as np

a = np.load('/Users/matteo/Downloads/data/drug_data.npz')
print('Drug data - authors')
print(a.files)
authors_X_train = a['X_train']
authors_X_test = a['X_test']
authors_Y_train = a['Y_train']
authors_Y_test = a['Y_test']
print(f"Shape of THEIR data:\n  -X_train: {authors_X_train.shape}\n  -X_test: {authors_X_test.shape}\n  -Y_train: {authors_Y_train.shape}\n  -Y_test: {authors_Y_test.shape}\n")


b = np.load('/Users/matteo/Dropbox/Università/Module 3/FACT/Project/FACT-AI/Custom data preprocessing/processed/drug_data.npz')
print('Drug data - custom')
print(b.files)
our_X_train = b['X_train']
our_X_test = b['X_test']
our_Y_train = b['Y_train']
our_Y_test = b['Y_test']
print(f"Shape of OUR data:\n  -X_train: {our_X_train.shape}\n  -X_test: {our_X_test.shape}\n  -Y_train: {our_Y_train.shape}\n  -Y_test: {our_Y_test.shape}\n")

print(f"Number of 1's in THEIR Y_train: {np.sum(authors_Y_train)}, Y_test: {np.sum(authors_Y_test)}, Tot: {np.sum(authors_Y_train) + np.sum(authors_Y_test)}")
print(f"Number of 1's in OUR   Y_train: {np.sum(our_Y_train)}, Y_test: {np.sum(our_Y_test)}, Tot: {np.sum(our_Y_train) + np.sum(our_Y_test)}")

# print("Authors:-------------\n", authors_group_label[:10,:6])
# print("\nCustom:-------------\n", our_group_label[:10,:6])

# for col in range(authors_X_train.shape[1]):
#     print(f"Column {col}:\n")
#     print(np.unique(authors_X_train[:,col]))
#     print(np.unique(our_X_train[:,col]))
#     print(f"{np.array_equal(np.unique(authors_X_train[:,col]).astype(np.float32), np.unique(our_X_train[:,col]).astype(np.float32))}\n========================\n")
