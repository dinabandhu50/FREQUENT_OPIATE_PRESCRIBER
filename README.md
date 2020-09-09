# FREQUENT OPIATE PRESCRIBER

The aim of this project is to reduce the death due to drug over dose. Here we are determining who is prescribing opiate very frequently in USA. We classify the prescriber frequent if he is prescribing more the 10 opiate associated drug within a year.

## How to use it

1. I have used data from kaggle site [This Link](https://www.kaggle.com/apryor6/us-opiate-prescriptions)

2. Set up conda env:

    1. create an env
        > `conda create -n fop`
    2. install all required packages
        > `conda env update -f env.yml`

3. Notebooks:

    1. When I was implementing this project, I used notebooks. I have them in two files.
    2. In **_EDA.ipynb_** I have all my Data analysis, and in **_ml_model.ipynb_** I have written all the training codes.
    3. The **_config.py_** has all hardcoded variable values.
    4. The **_pipeline.py_** has the main pipeline
    5. The **_preprocessors.py_** has all the custom preprocessors.

4. src folder:
    1. src has all the python scripts for deployment easy style.
