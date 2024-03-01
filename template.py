from pathlib import Path
import os

# Specifying the list of file and folders that required
package_name = "CreditCardFraudDetection"
list_of_files = [
    ".github/workflows/main.yaml",
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/components/data_ingestion.py",
    f"src/{package_name}/components/data_transformation.py",
    f"src/{package_name}/components/model_training.py",
    f"src/{package_name}/pipelines/__init__.py",
    f"src/{package_name}/pipelines/training_pipeline.py",
    f"src/{package_name}/pipelines/prediction_pipeline.py",
    f"src/{package_name}/logger.py",
    f"src/{package_name}/exception.py",
    f"src/{package_name}/utils/__init__.py",
    "notebooks/research.ipynb",
    "notebooks/data/.gitkeep",
    "notebooks/analysis.ipynb",
    "requirements.txt",
    "setup.py",
    "init_setup.sh",
    "app.py",
    "templates/form.html",
    "README.md"
]

# Here will create a directory
for filepath  in list_of_files:
    filepath = Path(filepath)
    file_dir, file_name = os.path.split(filepath)

    # Create directories for which there are values in file directory
    # 'exist_ok=True' : If directory already exist, this command will avoid raising error. 
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)


    # create files if not exist
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,"w") as f:
            pass
    else:
        print("file already exists")
        
                