import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


folder_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data"

def create_window_labels(pat_labels):
    has_one_in_row = np.any(pat_labels == 1, axis=1)
    has_one_in_row = has_one_in_row.astype(int)
    return has_one_in_row




def countNumberOfSeizuresPerPerson(patient_number):
    # Iterate through all files in the folder
    cnt = 0
    for filename in os.listdir(folder_path):
        # Check if the file starts with "pat_1" and has the ".npy" extension
        if filename.startswith(f"pat_{patient_number}_"):
            # Construct the full path to the file

            cnt += 1
    return cnt //4

def sensitiyity(generated_labels , true_labels):
    TP = 0
    FN = 0
    if(generated_labels.shape[0] != true_labels.shape[0]):  # check if the number of labels is the same
        print("The number of labels is not the same")
        return
    else:
        true_labels_size = true_labels.shape[0]
        for i in range(0, true_labels_size):
            if generated_labels[i] == 1 and true_labels[i] == 1:
                TP += 1
            if generated_labels[i] == 0 and true_labels[i] == 1:
                FN += 1
        a = {"sens": TP/(TP+FN) ,"TP": TP, "FN": FN}
        return a

def specificity(generated_labels , true_labels):
    TN = 0
    FP = 0
    if(generated_labels.shape[0] != true_labels.shape[0]):  # check if the number of labels is the same
        print("The number of labels is not the same")
        return
    else:
        true_labels_size = true_labels.shape[0]

        for i in range(0, true_labels_size):
            if generated_labels[i] == 0 and true_labels[i] == 0:
                TN += 1
            if generated_labels[i] == 1 and true_labels[i] == 0:
                FP += 1
        a = {"spec": TN/(TN+FP), "TN": TN, "FP": FP}
        return a


path_extension_ll = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\filtered_data_romina"
path_extension_labels = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels"
thresholds = [0.65, 1,2,3]

list_th_0 = []

for th in thresholds:
    th_list = []
    for p in range(1, 2):
        ch_num = 1
        sz_num = countNumberOfSeizuresPerPerson(p)
        innerList = []
        for i in range(1, 2):

            file_path_ll = os.path.join(path_extension_ll, f"ll_pat_{p}_sz_{i}_ch_{ch_num}.npy")
            ll = np.load(file_path_ll)
            new_ll = ll[1, :]


            file_path_labels = os.path.join(path_extension_labels, f"pat_{p}_sz_{i}_labels.npy")
            ##df = pd.read_csv(file_path_labels)
            ##condition = df.iloc[:, 0] == 1
            ##label = df[condition]
            label_np = np.load(file_path_labels)
            label_np_1 = create_window_labels(label_np)

            print("--------------------")
            print(new_ll.shape)
            print(label_np_1.shape)


            new_ll_label = (new_ll > th).astype(int)
            sens = sensitiyity(new_ll_label, label_np_1)

            spec = specificity(new_ll_label, label_np_1)
            innerList.append([1 - spec["spec"],sens["sens"]])

            # Assuming you have y_true (true labels) and y_scores (predicted scores or probabilities)
            fpr, tpr, thresholds = roc_curve(label_np_1, new_ll)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
        plt.show()

            #plt.plot(innerList)



#
#
#
#
#
#
# df = pd.read_csv(r"/masterThesis/real_data/labels_s/pat_20_sz_1.csv")
#
# condition = df.iloc[:,0] == 1
# label = df[condition]
# label_np_1 = label["label"].to_numpy()
#
#
# ll = np.load(r"/masterThesis/real_data/filtered_data_romina/ll_pat_20_sz_1_ch_1.npy")
# new_ll = ll[1, :]
# old_ll = ll[0, :]
#
#
#
# avg_new = np.average(new_ll)
# std_old = np.std(new_ll)
#
# avg_old = np.average(old_ll)
# std_old = np.std(old_ll)
#
#
#
# i_range = [0, 0.5, 1, 1.5, 2, 2.5, 3]
#
# thresholds_new = []
# for i in i_range:
#     thresholds_new.append(avg_new + i * std_old)
#     thresholds_new.append(avg_new - i * std_old)
#
# thresholds_old = []
# for i in i_range:
#     thresholds_old.append(avg_old + i * std_old)
#     thresholds_old.append(avg_old - i * std_old)
#
#
#
#
# for th in thresholds_new:
#     new_ll_label = (new_ll > th).astype(int)
#     sens = sensitiyity(new_ll_label, label_np_1)
#
#     spec = specificity(new_ll_label, label_np_1)
#     print("new metrics")
#     print(f"number of true zeros: {np.count_nonzero(label_np_1 == 0)}, number of true ones: {np.count_nonzero(label_np_1 == 1)}")
#     print(f"th: {th}, sens: {sens['sens']}, spec: {spec['spec']}, TP: {sens['TP']}, FN: {sens['FN']}, TN: {spec['TN']}, FP: {spec['FP']}")
# print("---------------------------------------------------")
# for th in thresholds_old:
#     old_ll_label = (old_ll > th).astype(int)
#     sens = sensitiyity(old_ll_label, label_np_1)
#     spec = specificity(old_ll_label, label_np_1)
#     print("old metrics")
#     print(f"number of true zeros: {np.count_nonzero(label_np_1 == 0)}, number of true ones: {np.count_nonzero(label_np_1 == 1)}")
#     print(f"th: {th}, sens: {sens['sens']}, spec: {spec['spec']}, TP: {sens['TP']}, FN: {sens['FN']}, TN: {spec['TN']}, FP: {spec['FP']}")
#     # print(f"th: {th}, sens: {sens[0]}, spec: {spec{"sepc"}}")