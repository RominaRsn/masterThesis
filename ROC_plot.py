import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedKFold


folder_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data"
ll_path_bw = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\bw_filter\ll"
def ROCParamsForBWFiltering_30(p, sz_num, ch_num, true_labels):
    extension = f"ll_pat_{p}_sz_{sz_num}_ch_{ch_num}.npy"
    ll_bw = np.load(os.path.join(ll_path_bw, extension))
    ll_bw = ll_bw[0, :]

    avg = np.average(ll_bw)
    std = np.std(ll_bw)
    thresholds = [avg - 3 * std, avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * old_std]

    sens = []
    spec = []
    for th in thresholds:
        new_ll_label = (ll_bw > th).astype(int)
        sens1 = sensitivity(new_ll_label, true_labels)
        spec1 = specificity(new_ll_label, true_labels)
        sens.append(sens1["sens"])
        spec.append(1 - spec1["spec"])

    return sens, spec


def ROCParamsForBWFiltering_45(p, sz_num, ch_num, true_labels):
    extension = f"ll_pat_{p}_sz_{sz_num}_ch_{ch_num}.npy"
    ll_bw = np.load(os.path.join(ll_path_bw, extension))
    ll_bw = ll_bw[1, :]

    avg = np.average(ll_bw)
    std = np.std(ll_bw)
    thresholds = [avg - 3 * std, avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * old_std]

    sens = []
    spec = []
    for th in thresholds:
        new_ll_label = (ll_bw > th).astype(int)
        sens1 = sensitivity(new_ll_label, true_labels)
        spec1 = specificity(new_ll_label, true_labels)
        sens.append(sens1["sens"])
        spec.append(1 - spec1["spec"])

    return sens, spec


def ROCParamsForBWFiltering_70(p, sz_num, ch_num, true_labels):
    extension = f"ll_pat_{p}_sz_{sz_num}_ch_{ch_num}.npy"
    ll_bw = np.load(os.path.join(ll_path_bw, extension))
    ll_bw = ll_bw[2, :]

    avg = np.average(ll_bw)
    std = np.std(ll_bw)
    thresholds = [avg - 3 * std, avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * old_std]

    sens = []
    spec = []
    for th in thresholds:
        new_ll_label = (ll_bw > th).astype(int)
        sens1 = sensitivity(new_ll_label, true_labels)
        spec1 = specificity(new_ll_label, true_labels)
        sens.append(sens1["sens"])
        spec.append(1 - spec1["spec"])

    return sens, spec









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

def sensitivity(generated_labels , true_labels):
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


# path_extension_ll = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\new_filtered"
path_extension_labels = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels_s"
path_extension_ll = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cnn_combo\ll"


list_th_0 = []

for p in range(10,20):
    sz_num = countNumberOfSeizuresPerPerson(p)
    innerList = []
    for i in range(1, sz_num + 1):
        fig, axs = plt.subplots(1, 1)
        fig.suptitle(f"Patient {p} - seizure {i}")
        for ch_num in range(1,2):
            file_path_ll = os.path.join(path_extension_ll, f"ll_combo_pat_{p}_sz_{i}_ch_{ch_num}.npy")
            ll = np.load(file_path_ll)
            new_ll = ll[1, :]
            old_ll = ll[0, :]


            file_path_labels = os.path.join(path_extension_labels, f"pat_{p}_sz_{i}_labels.npy")
            ##df = pd.read_csv(file_path_labels)
            ##condition = df.iloc[:, 0] == 1
            ##label = df[condition]
            label = np.load(file_path_labels)

            print("--------------------")
            print(new_ll.shape)
            print(label.shape)

            # Creating Thresholds
            old_avg = np.average(old_ll)
            old_std = np.std(old_ll)
            thresholds_old = [old_avg - 3 * old_std, old_avg - 2 * old_std, old_avg - old_std, old_avg, old_avg + old_std, old_avg + 2 * old_std, old_avg + 3 * old_std]

            avg = np.average(new_ll)
            std = np.std(new_ll)
            thresholds = [avg - 3 * std, avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * std]
            # for th in thresholds:
            #     new_ll_label = (new_ll > th).astype(int)
            #     fpr, tpr, thresholds = roc_curve(label, new_ll_label)
            #     roc_auc = auc(fpr, tpr)
            #     plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            #
            #
            #
            # for th in thresholds_old:
            #     old_ll_label = (old_ll > th).astype(int)
            #     fpr, tpr, thresholds = roc_curve(label, old_ll_label)
            #     roc_auc = auc(fpr, tpr)
            #     plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            #
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic (ROC) Curve')
            # plt.legend(loc='lower right')
            # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            # plt.show()

            # Create subplots
            #fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

            # Plot ROC curves for 'new_ll'
            sens_new = []
            spec_new = []
            #axs[0].set_title('ROC Curve for new_ll')
            #axs[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            for th in thresholds:
                new_ll_label = (new_ll > th).astype(int)
                sens1 = sensitivity(new_ll_label, label)
                spec1 = specificity(new_ll_label, label)
                sens_new.append(sens1["sens"])
                spec_new.append(1 - spec1["spec"])




                # fpr, tpr, _ = roc_curve(label, new_ll_label)
                # roc_auc = auc(fpr, tpr)



            # Plot ROC curves for 'old_ll'


            sens_old = []
            spec_old = []
            for th in thresholds_old:
                old_ll_label = (old_ll > th).astype(int)
                sens2 = sensitivity(old_ll_label, label)
                spec2 = specificity(old_ll_label, label)
                sens_old.append(sens2["sens"])
                spec_old.append(1 - spec2["spec"])


            auc_old = auc(spec_old, sens_old)
            axs.set_title('ROC Curve for old_ll')
            axs.plot(spec_old, sens_old, lw=2, label=f'ROC curve old (AUC = {auc_old:.2f})')

            axs.set_xlabel('False Positive Rate')
            axs.set_ylabel('True Positive Rate')


            auc_new = auc(spec_new, sens_new)
            # axs[1].set_title('ROC Curve for new_ll')
            axs.plot(spec_new, sens_new, lw=2, label=f'ROC curve new (AUC = {auc_new:.2f})')
            # axs[1].set_xlabel('False Positive Rate')
            # axs[1].set_ylabel('True Positive Rate')
            # axs[1].legend(loc='lower right')
            axs.legend(loc='lower right')


            print("auc_old")
            print(auc_old)
            print("auc_new")
            print(auc_new)

            # sens_30, spec_30 = ROCParamsForBWFiltering_30(p, i, ch_num, label)
            # auc_30 = auc(spec_30, sens_30)
            # axs[ch_num -1].plot(spec_30, sens_30, label=f'ROC curve for bw-30hz (AUC = {auc_30:.2f})')
            # sens_45, spec_45 = ROCParamsForBWFiltering_45(p, i, ch_num, label)
            # auc_45 = auc(spec_45, sens_45)
            # axs[ch_num-1].plot(spec_45, sens_45, label=f'ROC curve for bw-45 hz (AUC = {auc_45:.2f})')
            # sens_70, spec_70 = ROCParamsForBWFiltering_70(p, i, ch_num, label)
            # auc_70 = auc(spec_70, sens_70)
            # axs[ch_num-1].plot(spec_70, sens_70, label=f'ROC curve for bw-70hz (AUC = {auc_70:.2f})')
            #
            # axs[ch_num-1].plot(spec_new, sens_new, label=f'ROC curve for procesed data (AUC = {auc_new:.2f})')
            # axs[ch_num-1].plot(spec_old, sens_old, label=f'ROC curve for unprocessed data (AUC = {auc_old:.2f})')
            # #plt.legend(["bw-30hz","bw-45hz", "bw-70hz","processed", "unprocessed"])
            # axs[ch_num-1].set_xlabel('1 - Specificity')
            # axs[ch_num-1].set_ylabel('Sensitivity')
            # axs[ch_num-1].set_title(f'ROC Curve for channel {ch_num}')
            # axs[ch_num-1].legend(loc ="lower right")

        plt.show()
            # fpr, tpr, _ = roc_curve(label, old_ll_label)
            # roc_auc = auc(fpr, tpr)
            # axs[1].plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        #
        # axs[1].set_xlabel('False Positive Rate')
        # axs[1].set_ylabel('True Positive Rate')
        # axs[1].legend(loc='lower right')

        # # Adjust layout to prevent clipping of titles
        # plt.tight_layout()
        #
        # # Show the plot
        # plt.show()


        #     sens = sensitiyity(new_ll_label, label_np_1)
        #
        #     spec = specificity(new_ll_label, label_np_1)
        #     innerList.append([1 - spec["spec"],sens["sens"]])
        #
        #     # Assuming you have y_true (true labels) and y_scores (predicted scores or probabilities)
        #     fpr, tpr, thresholds = roc_curve(label_np_1, new_ll)
        #     roc_auc = auc(fpr, tpr)
        #
        #     # Plot ROC curve
        #     plt.figure(figsize=(8, 8))
        #     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        #     plt.xlabel('False Positive Rate')
        #     plt.ylabel('True Positive Rate')
        #     plt.title('Receiver Operating Characteristic (ROC) Curve')
        #     plt.legend(loc='lower right')
        # plt.show()

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