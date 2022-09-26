# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Customer churn is the percentage of customers that stopped using a company's product or service during a certain time frame.

This project is predicting churn for credit card customers of a bank. To predict the customer churn the code utilizes machine learning models using which it can predict and assign a predicted churn flag of (0,1) to a customer. 

Using 2 different models to find which model suits the best to data & results in a better prediction.

Models Used: Random Forest Classifier
           : Logistic Regression

After running the script, A user will be able to get the predicted customer churn for the test customer data and finally will be 
able to compare the predicted results to the original results using metrices like classification report, etc. 

## Files and data description
### Folders
DATA : Folder that contains the bank data file i.e  bank_data.csv
Images: This folder contains 2 sub-folders i.e EDA & RESULTS.
Logs: Folder that will have a .log file which contains all metadata recorded while running the script
Models: Folder to save the resultant ml models in a .pkl file.

### Files
Churn_library: Predict customer churn package to predict the customer churn using ML.
churn_notebook: Interactive version of Predict customer churn package.
churn_script_logging_and_tests: Testing and logging file to test the main package functions & log the errors and metadata to the .log file in ./Logs/
Requirements: Dependency requirement file to install required dependencies.

### Data description

No of columns : 22
Rows in data: 10127
Description of Data: ![alt text](https://drive.google.com/file/d/1MbMjMR6hSFhp0aTf2eh73fUCM6XkVdct/view?usp=sharing)

## Running Files

### Churn package
Churn_library package can be run in both interactive mode & command line mode. 

1)To run the file in interactive mode, bring the code from churn_library.py in a .ipynb file cell(jupyter notebook) and execute via: **Tab Run>Run all cells**. 

2)To run the same file in cmd, go to the project folder's location in cmd then type command: **python churn_library.py** or **ipython churn_library.py**

After the script ran successfully, user can view the results,plots and models in their respective locations.

### Churn script logging and testing
Churn_script_logging_and_tests is a testing file to test all edge cases of function to make sure the churn script is running the way it's supposed to run. File contains a series of test functions which uses the churn package functions as fixtures to test them. 

To run the churn_script_logging_and_tests.py file, Open command prompt(cmd) then go to the project folder location using **cd folder_name** command then type command: **python churn_script_logging_and_tests.py** or **ipython churn_script_logging_and_tests.py**.

User can check the logs and browse the .log file to view the metadata and errors(if any).

Best Practices: Run the churn_script_logging_and_tests file before running the churn_package. 