import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")




# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)



#result_3layer = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_three_layers_result_moreEpoch.npy")
result_3layer_more_filters = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_three_layers_result_morefilters.npy")
result_4layer = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_four_layers_results.npy")
result_5layer = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_results.npy")

result_5layer_more_filter = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_results_more_filters.npy")
#, sharey='col'
signalIndexVector = [0, 1, 3, 7, 11, 13, 14, 16, 17]
for i in signalIndexVector:
    fig, axes = plt.subplots(nrows=5 , ncols=1)

    row_index = i
    #row_index = np.random.randint(0, a)
    #col_index = np.random.randint(0, 11520000/500)

    axes[0].plot(clean_test[row_index, :], label = 'Clean Data')
    axes[0].set_title('Clean data')
    axes[0].set_ylabel('Signal amplitude')
    axes[0].set_xlabel('Time')

    #print(smaller_reshaped_data_clean_test[row_index, :].shape)


    axes[1].plot(noisy_test[row_index, :], label = 'Noisy Data')
    axes[1].set_title('Noisy data')
    axes[1].set_ylabel('Signal amplitude')
    axes[1].set_xlabel('Time')

    #result = model.predict(result)
    #result = result.transpose()

    axes[2].plot(result_3layer_more_filters[row_index, :] - .55, label='predicted data with 3 layers in encoder')
    axes[2].set_title('predicted data with 3 layers in encoder')
    axes[2].set_ylabel('Signal amplitude')
    axes[2].set_xlabel('Time')


    # axes[3].plot(result_4layer[row_index, :] - .55, label ='predicted data with 4 layers in encoder')
    # axes[3].set_title('predicted data with 4 layers in encoder')
    # axes[3].set_ylabel('Signal amplitude')
    # axes[3].set_xlabel('Time')

    axes[3].plot(result_5layer[row_index, :] - .55, label ='predicted data with 5 layers in encoder')
    axes[3].set_title('predicted data with 5 layers in encoder')
    axes[3].set_ylabel('Signal amplitude')
    axes[3].set_xlabel('Time')

    axes[4].plot(result_5layer_more_filter[row_index, :], label ='predicted data with 5 layers in encoder with more filters')
    axes[4].set_title('predicted data with 5 layers in encoder with more filters')
    axes[4].set_ylabel('Signal amplitude')
    axes[4].set_xlabel('Time')



    #test_array = np.array(noisy_dataF3[row_index, col_index : col_index + 500])
    #print(test_array.shape())

    # Add overall title
    fig.suptitle('Comparison of different encoders', fontsize=16)

    # Adjust layout to prevent overlap
    #plt.tight_layout()

    # Show the plot
    plt.show()