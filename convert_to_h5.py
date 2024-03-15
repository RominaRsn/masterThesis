import os

# Specify the paths to the input and output files
keras_file = r'C:\Users\RominaRsn\Downloads\model_with_5_layers_paper_arch_EMG_EOG.keras'
output_h5_file = r'C:\Users\RominaRsn\Downloads\out.h5'

# Open the .keras file in binary mode and read its contents
with open(keras_file, 'rb') as f:
    # Read the content of the .keras file
    keras_content = f.read()

    # Find the start and end indices of the HDF5 content within the .keras file
    start_index = keras_content.find(b'\x89HDF\r\n\x1a\n')
    end_index = keras_content.find(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')

    # Extract the HDF5 content
    h5_content = keras_content[start_index:end_index]

    # Write the extracted HDF5 content to a new .h5 file
    with open(output_h5_file, 'wb') as h5_file:
        h5_file.write(h5_content)

print("Extraction complete.")