# Complex Wilson Loop - quark anti-quark potential project

Machine learning project to use network setup to learn the deformation parameters in our deformed AdS-BH model metric. 

## Setup

1. Create virtual environment: 
```bash
python3 -m venv .venv
source .venv/bin/activate
````

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data

Data files should be placed in the `data/`directory (our data is not public so it will not display here in the repository)

## Usage

Main training notebook: `MAIN-data-training-HR.ipynb`

## Structure 

- `dataset_HR.py`- Module for dataset handling
- `constants.py`- Module for the dtype constants
- `model_HR_new.py`- Module for the training model
- `check_integrals.py`- Integral verification for the post-processing notebooks 
