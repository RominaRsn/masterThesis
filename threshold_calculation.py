import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def sensitiyity(generated_labels , true_labels):
    TP = 0
    FN = 0
    for i in range(len(generated_labels)):
        if generated_labels[i] == 1 and true_labels[i] == 1:
            TP += 1
        if generated_labels[i] == 0 and true_labels[i] == 1:
            FN += 1
    a = {"sens": TP/(TP+FN) ,"TP": TP, "FN": FN}
    return a

def specificity(generated_labels , true_labels):
    TN = 0
    FP = 0
    for i in range(len(generated_labels)):
        if generated_labels[i] == 0 and true_labels[i] == 0:
            TN += 1
        if generated_labels[i] == 1 and true_labels[i] == 0:
            FP += 1
    a = {"spec": TN/(TN+FP), "TN": TN, "FP": FP}
    return a


df = pd.read_csv(r"/masterThesis/real_data/labels_s/pat_20_sz_1.csv")

condition = df.iloc[:,0] == 1
label = df[condition]
label_np_1 = label["label"].to_numpy()


ll = np.load(r"/masterThesis/real_data/filtered_data_romina/ll_pat_20_sz_1_ch_1.npy")
new_ll = ll[1, :]
old_ll = ll[0, :]



avg_new = np.average(new_ll)
std_old = np.std(new_ll)

avg_old = np.average(old_ll)
std_old = np.std(old_ll)



i_range = [0, 0.5, 1, 1.5, 2, 2.5, 3]

thresholds_new = []
for i in i_range:
    thresholds_new.append(avg_new + i * std_old)
    thresholds_new.append(avg_new - i * std_old)

thresholds_old = []
for i in i_range:
    thresholds_old.append(avg_old + i * std_old)
    thresholds_old.append(avg_old - i * std_old)




for th in thresholds_new:
    new_ll_label = (new_ll > th).astype(int)
    sens = sensitiyity(new_ll_label, label_np_1)

    spec = specificity(new_ll_label, label_np_1)
    print("new metrics")
    print(f"number of true zeros: {np.count_nonzero(label_np_1 == 0)}, number of true ones: {np.count_nonzero(label_np_1 == 1)}")
    print(f"th: {th}, sens: {sens['sens']}, spec: {spec['spec']}, TP: {sens['TP']}, FN: {sens['FN']}, TN: {spec['TN']}, FP: {spec['FP']}")
print("---------------------------------------------------")
for th in thresholds_old:
    old_ll_label = (old_ll > th).astype(int)
    sens = sensitiyity(old_ll_label, label_np_1)
    spec = specificity(old_ll_label, label_np_1)
    print("old metrics")
    print(f"number of true zeros: {np.count_nonzero(label_np_1 == 0)}, number of true ones: {np.count_nonzero(label_np_1 == 1)}")
    print(f"th: {th}, sens: {sens['sens']}, spec: {spec['spec']}, TP: {sens['TP']}, FN: {sens['FN']}, TN: {spec['TN']}, FP: {spec['FP']}")
    # print(f"th: {th}, sens: {sens[0]}, spec: {spec{"sepc"}}")