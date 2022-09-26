'''

Package: Machine learning based customer churn prediction
Author: Prashant Singh
Date: September, 2022

'''

# import libraries
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_roc_curve
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    Returns dataframe for the csv file found at pth i.e path

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    df_ = pd.read_csv(pth)
    return df_


def perform_eda(df_data):
    '''
    Perform eda on df and save figures to images folder

    Input: df: pandas dataframe

    Output: None
    '''
    #encoding categorical column to a flag(0,1)
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df_data['Churn'].hist()
    plt.xlabel('Churn')
    plt.savefig(r'.\Images\eda\churn_plot.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    df_data['Customer_Age'].hist()
    plt.xlabel('Customer_age')
    plt.savefig(r'.\Images\eda\customer_age_plot.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    df_data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.xlabel('Marital_Status')
    plt.savefig(r'.\Images\eda\marital_status_plot.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.histplot(df_data['Total_Trans_Ct'], stat='density', kde=True)
    plt.xlabel('Total_Trans_Ct')
    plt.savefig(r'.\Images\eda\total_trans_Ct_plot.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(df_data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(r'.\Images\eda\heatmap_corr.png')
    plt.close()


def encoder_helper(df_data, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns each having a proportion
            of churn for their respective category
    '''
    #encoding categorical columns
    for col in category_lst:
        groups = dict(df_data.groupby(col).mean()[response])
        df_data[col + '_' + response] = df_data[col].apply(lambda x: groups[x])
    return df_data


def perform_feature_engineering(df_data, response='Churn'):
    '''
    Feature engineering function to split the whole dataset into 2 subsets:
    training & testing datasets.
    Training data: Data used to train the model
    Testing data: Data used to make predictions

    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for
              naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    #columns to keep in X data
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
                 'Income_Category_Churn', 'Card_Category_Churn']
    
    x_data = df_data[keep_cols]
    y_data = df_data[response]
    # splitting the data in train & testing sets
    x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)
    return x_train_df, x_test_df, y_train_df, y_test_df


def classification_report_image(ytrain,
                                ytest,
                                ytrain_preds_lr,
                                ytrain_preds_rf,
                                ytest_preds_lr,
                                ytest_preds_rf):
    '''
    produces classification report for training and testing results
    and stores report as image in images folder
    
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                ytest, ytest_preds_rf)), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                ytrain, ytrain_preds_rf)), {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(r'.\Images\results\rfc_test_train_report.png')
    plt.close()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                ytrain, ytrain_preds_lr)), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                ytest, ytest_preds_lr)), {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(r'.\Images\results\lr_test_train_report.png')
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    x_train, x_test = train_test_split(x_data, test_size=0.3, random_state=42)
    #Uses Tree SHAP algorithms to explain the output of ensemble tree model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False, plot_size=(19, 19))
    plt.savefig(output_pth + 'shap_tree.png')
    plt.close()

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    plt.figure(figsize=(20, 25))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=75)
    plt.savefig(output_pth + 'feature_importances.png')
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    #Creating instances of ml models
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    
    # parameters dictionary to find the best set of parameters for rfc model
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    
    #exhaustive search over parameter values for rfc estimator
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    
    #training models
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, r'.\Models\rfc_model.pkl')
    joblib.dump(lrc, r'.\Models\logistic_Model.pkl')

    # save roc_curves
    plt.figure(figsize=(20, 10))
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    plt.savefig(r'.\Images\results\roc_curve_lrc_rfc1.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    axis = plt.gca()
    plot_roc_curve(cv_rfc, x_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig(r'.\Images\results\roc_curve_lrc_rfc2.png')
    plt.close()


if __name__ == "__main__":
    # importing data from csv
    DF = import_data(r'.\Data\bank_data.csv')
    # Performing eda on data & saving the results to images/eda folder
    perform_eda(DF)

    # Feature encoding various categorical variables in the dataset
    encoder_helper(DF, ['Education_Level', 'Marital_Status',
                        'Gender', 'Income_Category', 'Card_Category'])

    # unpacking the results
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(DF)

    # training models using the X,y datasets
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

    # Loading best tuned models
    RFC_MODEL = joblib.load(r'.\Models\rfc_model.pkl')
    LR_MODEL = joblib.load(r'.\Models\logistic_model.pkl')

    # Predicting results on train & test set using rfc_model
    Y_TRAIN_PREDS_RF = RFC_MODEL.predict(X_TRAIN)
    Y_TEST_PREDS_RF = RFC_MODEL.predict(X_TEST)

    # Predicting results on train & test set using lr_model
    Y_TRAIN_PREDS_LR = LR_MODEL.predict(X_TRAIN)
    Y_TEST_PREDS_LR = LR_MODEL.predict(X_TEST)

    # Storing classification reprot
    classification_report_image(Y_TRAIN, Y_TEST, Y_TRAIN_PREDS_LR,
                                Y_TRAIN_PREDS_RF,
                                Y_TEST_PREDS_LR,
                                Y_TEST_PREDS_RF)
    # Creating and storing feature importances plots
    X_DATA = pd.concat([X_TRAIN, X_TEST])
    feature_importance_plot(RFC_MODEL, X_DATA, output_pth=r'./Images/results/')
