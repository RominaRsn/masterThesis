import subprocess

# List of Python files to run
files_to_run = ['metrics_calculation_eog.py', 'metrics_calculation_emg.py', 'comparing_encoders.py', 'paper_cnn_train.py', 'model_with_7_layers_train.py']

# Loop through the list and run each file
for file in files_to_run:
    try:
        subprocess.run(['python', file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {file}: {e}")