from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



# Create a sample dataset (replace this with your data)
# np.random.seed(42)
# data = np.random.rand(100, 5)  # 100 samples with 5 features each
data = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\filtered_data_romina\ll_pat_2_sz_1_ch_2.npy")


df = pd.read_csv(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels_s\pat_2_sz_1.csv")

condition = df.iloc[:,0] == 1
label = df[condition]
label_np_1 = label["label"].to_numpy()


# data = np.transpose(data)
# # Specify the number of components you want to keep (reduce to)
# n_components = 1
#
# # Instantiate the PCA model
# pca = PCA(n_components=n_components)
#
# # Fit the PCA model to the data and transform the data
# transformed_data = pca.fit_transform(data)
#
# # Print the explained variance ratio for each principal component
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)
#
# # Print the transformed data
# print("\nTransformed Data:")
# print(transformed_data.shape)
#
# # Plot the transformed data
# plt.figure(figsize=(8, 6))
# plt.plot(transformed_data[:, 0])
# plt.title('PCA: Scatter Plot of Transformed Data')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True)
# plt.show()

# plt.plot(data[0, :])
# plt.plot(data[1, :])
# plt.plot(label_np_1, 'r')
# plt.legend(['data', 'predicted data', 'label'])
# plt.show()




#plotting labels
folder_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data"

def countNumberOfSeizuresPerPerson(patient_number):
    # Iterate through all files in the folder
    cnt = 0
    for filename in os.listdir(folder_path):
        # Check if the file starts with "pat_1" and has the ".npy" extension
        if filename.startswith(f"pat_{patient_number}_"):
            # Construct the full path to the file

            cnt += 1
    return cnt
p_range = [20, 21, 40]
for p in p_range:
        sz_num = countNumberOfSeizuresPerPerson(p) //4
        for i in range(1, 3):
            label_path = os.path.join(folder_path, f"labels_s\pat_{p}_sz_{i}.csv")
            df = pd.read_csv(label_path)
            condition = df.iloc[:,0] == 1
            label = df[condition]
            label_np_1 = label["label"].to_numpy()
            plt.plot(label_np_1)
            plt.legend([f'label{i}'])


plt.show()
