echo [${data}]: "START"

echo [${data}]: "CREATING VENV WITH 3.9 PYTHON VERSION"

# conda create --prefix ./venv_13 python==3.9 -y
conda create -p ./venv python==3.9 -y

echo [${data}]: "ACTIVATING THE ENVIRONMENT"

conda activate ./venv

echo [${data}]: "INSTALLING LIBRARIES FROM requirements.txt"

pip install -r requirements.txt

echo [${data}]: "END"