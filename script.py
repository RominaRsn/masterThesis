import subprocess

# List of Python files to run
files_to_run = [r"C:\Users\Romina\masterThesis\final_model_five_layer.py", r"C:\Users\Romina\masterThesis\final_model_threee_layer.py", r"C:\Users\Romina\masterThesis\model_with_GRU.py"]

# Loop through the list and run each file
for file in files_to_run:
    try:
        subprocess.run(['python', file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {file}: {e}")