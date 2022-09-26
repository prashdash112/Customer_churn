'''

Package: Customer churn testing and logging file
Author: Prashant Singh
Date: September, 2022

'''

import os
import logging
import joblib
from churn_library import import_data
from churn_library import perform_eda
from churn_library import encoder_helper
from churn_library import perform_feature_engineering
from churn_library import train_models

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    logging.info("########################################################")
    logging.info("Testing import_data")
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Shape of df:{}".format(df.shape))
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    logging.info("Testing perform eda")
    try:
        df = import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        perform_eda(df)
        assert os.path.isfile("./images/eda/churn_plot.png")
        assert os.path.isfile("./images/eda/customer_age_plot.png")
        assert os.path.isfile("./images/eda/heatmap_corr.png")
        assert os.path.isfile("./images/eda/marital_status_plot.png")
        assert os.path.isfile("./images/eda/total_trans_Ct_plot.png")
        logging.info("Testing perform eda: SUCCESS")
        logging.info("The plot files are available at location: ./images/eda/")
    except AssertionError as err:
        logging.error(
            "Testing perform eda: The plot files doesn't appear at the location './images/eda'")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    logging.info("Testing encoder helper")
    try:
        df = import_data(r"./data/bank_data.csv")
        perform_eda(df)
        encoder_helper(df, ['Education_Level', 'Marital_Status',
                            'Gender', 'Income_Category', 'Card_Category'],
                       response='Churn')

        assert df['Education_Level_Churn'].shape[0] == len(df)
        assert df['Marital_Status_Churn'].shape[0] == len(df)
        assert df['Gender_Churn'].shape[0] == len(df)
        assert df['Income_Category_Churn'].shape[0] == len(df)
        assert df['Card_Category_Churn'].shape[0] == len(df)
        logging.info("Testing encoder helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder helper: Failed to encode the categorical columns")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    logging.info("Testing feature engineering function")
    try:
        df = import_data(r"./data/bank_data.csv")
        perform_eda(df)
        df_data = encoder_helper(df,
                                 ['Education_Level',
                                  'Marital_Status',
                                  'Gender',
                                  'Income_Category',
                                  'Card_Category'],
                                 response='Churn')
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_data, response='Churn')
        assert x_train.shape[0] != 0
        assert x_test.shape[0] != 0
        assert y_train.shape[0] != 0
        assert y_test.shape[0] != 0
        logging.info("Testing feature engineering function: SUCCESS")
        logging.info(
            "Shapes of x_train, x_test, y_train, y_test:{df1}{df2}{df3}{df4}".format(
                df1=x_train.shape,
                df2=x_test.shape,
                df3=y_train.shape,
                df4=y_test.shape))
    except AssertionError as err:
        logging.error(
            "ERROR: The train-test splitted data is either empty or inconsistent")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    logging.info("Testing train_models function")
    try:
        df = import_data(r"./data/bank_data.csv")
        perform_eda(df)
        df_data = encoder_helper(df,
                                 ['Education_Level',
                                  'Marital_Status',
                                  'Gender',
                                  'Income_Category',
                                  'Card_Category'],
                                 response='Churn')
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_data, response='Churn')
        train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile(r'.\Models\rfc_model.pkl')
        logging.info(
            "SUCCESS: RFC model is present at the location - ./Models/ ")
        rfc_model = joblib.load(r'.\Models\rfc_model.pkl')
        assert os.path.isfile(r'.\Models\logistic_model.pkl')
        logging.info(
            "SUCCESS: LFC model is present at the location - ./Models/ ")
        lr_model = joblib.load(r'.\Models\logistic_model.pkl')
        logging.info(
            "########################################################")

    except AssertionError as err:
        logging.error(
            "ERROR: Unable to train models and create model.pkl files at location - ./Models/ ")
        logging.info(
            "########################################################")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
