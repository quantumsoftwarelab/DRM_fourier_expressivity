# Constrained and Vanishing Expressivity of Quantum Fourier Models 
This code reproduces the figures in [Constrained and Vanishing Expressivity of Quantum Fourier Models](https://arxiv.org/abs/2403.09417). It implements the process of calculating empirical, analytic values and upper bounds of Fourier Coefficients for the Data Reuploading Model. 
## Installation
1. Clone the Repository
```
git clone https://github.com/quantumsoftwarelab/DRM_fourier_expressivity.git
cd DRM_fourier_expressivity
```
2. Set Up a Virtual Environment
Create and activate a Python virtual environment. This step is optional but recommended to maintain dependency isolation.

- On macOS and Linux:
```
python3 -m venv quantum_fourier_env
source quantum_fourier_env/bin/activate
```
- On Windows:

```
python -m venv quantum_fourier_env
.\\quantum_fourier_env\\Scripts\\activate
```
3. Install Dependencies
Install all the required packages listed in the requirements.txt file:

```
pip install -r requirements.txt
```
4. Verification
After installation, you can verify that the packages have been installed correctly by running:
```
pip list
```


## Tutorial
An example on how to run the functions is available in the [`main.ipynb`](/main.ipynb) notebook.

## Contact
For questions and suggestions, please contact Mario Herrero González: `mherrer3@ed.ac.uk`

## License
Distributed under the Apache-2.0 license. 
