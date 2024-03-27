import subprocess

# List of Python files to run
files_to_run = [r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\final_model_five_layers_true.py", r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\check_all_channels.py"]

# Loop through the list and run each file
for file in files_to_run:
    try:
        subprocess.run(['python', file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {file}: {e}")