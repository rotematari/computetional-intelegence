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