# Computational Intelligence Repository

This repository contains resources and code for computational intelligence practice.

## Getting Started

### Clone the Repository

To download this repository, use the following commands:

```bash
# Clone the repository
git clone https://github.com/rotematari/computetional-intelegence.git

# Navigate to the repository directory
cd computetional-intelegence
```

You can also download the repository as a ZIP file by clicking on the "Code" button on the GitHub page and selecting "Download ZIP".



# setup venv 

## Starting Fresh with pip

To set up a fresh environment with pip, follow these steps:

1. Create a new virtual environment:
    ```bash
    python -m venv venv
    ```

2. Activate the virtual environment:
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

3. Verify pip is installed:
    ```bash
    pip --version
    ```

4. Upgrade pip to the latest version:
    ```bash
    pip install --upgrade pip
    ```

5. If you need to start completely fresh, you can uninstall all packages:
    ```bash
    pip freeze > requirements.txt
    pip uninstall -r requirements.txt -y
    ```

6. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

7. When done, deactivate the virtual environment:
    ```bash
    deactivate
    ```

    ## Starting Fresh with Conda

    To set up a fresh environment with Conda, follow these steps:

    1. Install Miniconda or Anaconda if not already installed:
        - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
        - [Anaconda](https://www.anaconda.com/products/distribution)

    2. Create a new conda environment:
        ```bash
        conda create -n ci-env python=3.9
        ```

    3. Activate the conda environment:
        ```bash
        conda activate ci-env
        ```

    4. Install packages from conda-forge or pip:
        ```bash
        conda install -c conda-forge numpy pandas matplotlib
        # or use pip within the conda environment
        pip install -r requirements.txt
        ```

    5. Export your environment for sharing:
        ```bash
        conda env export > environment.yml
        ```

    6. When done, deactivate the conda environment:
        ```bash
        conda deactivate
        ```

    7. If needed, remove the environment:
        ```bash
        conda env remove -n ci-env
        ```