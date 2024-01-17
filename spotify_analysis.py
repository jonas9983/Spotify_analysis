from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import math
import numpy as np



def analyze_dataframe(features, filename, graph = 0):

    features = dataset_information(dataset = features, filename = filename, change = 0)

    data_exploration(dataset = features, show = [6,7])

    return

    # Logistic Regression with just one explanatory variable (numeric for now)
    
    y_test, y_pred, X_test, model = apply_logistic_regression_model(X = features.loc[:, features.columns.isin(['energy'])], 
                                                     y = features.loc[:, features.columns.isin(['danceability_binary'])])

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

    #sigmoid_function(model.coef_, model.intercept_)


    return


    # Logistic Regression with all the variables. It's so random

    y_test_wo_dummies, y_pred_wo_dummies = apply_logistic_regression_model(X =new_features.loc[:, ~new_features.columns.isin(['track_name', 'track_artist', 'key', 'genres', 'danceability_binary'])],
                                        y = new_features.loc[:, new_features.columns.isin(['danceability_binary'])])
    
    print("Accuracy: ", accuracy_score(y_test_wo_dummies, y_pred_wo_dummies))
    print("Confusion Matrix: \n", confusion_matrix(y_test_wo_dummies, y_pred_wo_dummies))
    print("Classification Report:\n", classification_report(y_test_wo_dummies, y_pred_wo_dummies))

    # Correlations between variables and danceability
    features_no_track_name = features.loc[:, ~features.columns.isin(['track_name', 'track_artist', 'key', 'genres'])]
    correlations = features_no_track_name.corr()
    #print(correlations['danceability_binary'])
    
    """print(features.head())
    print(features.describe().round(3))
    print(features[features.valence == features.valence.min()]) # Min valence
    print(features[features.valence == features.valence.max()]) # Max valence
    print(new_features.loc[:, ~new_features.columns.isin(['track_name', 'track_artist', 'key', 'genres'])].groupby('danceability_binary').mean())
"""

def apply_logistic_regression_model(X, y):
    # Logistic Regression 

    # Basic framework for logistic regression

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train,y_train.values.ravel())

    y_pred = model.predict(X_test)

    return(y_test, y_pred, X_test, model)

def dataset_information(dataset, filename, change = 0):
    dataset.head()
    dataset.columns
    dataset.info()


    dataset.isnull().sum() # There are no null values in the data
    dataset.track_name.nunique() # There are 487 unique values, but the dataset has 527 songs. This means that there are duplicates on the track_name column
    dataset.loc[dataset.duplicated()]
    # use dataset.query('track_name' == 'Name of the track') to check the duplicates 
    dataset[dataset.track_name.duplicated()]['track_name'].tolist()

    print(dataset.describe())

    if change == 1:
        ### Check if there are duplicate values on the track_name column
        dataset = dataset.drop_duplicates(subset = ['track_name'], keep = 'last').reset_index(drop=False).copy()
        dataset['danceability_binary'] = dataset['danceability'].apply(lambda x: int(1) if x >= 0.6 else int(0))

        ### Save new csv file 
        dataset.to_csv(filename, mode = 'w', index = False, header = True)

    print(dataset.info())

    return(dataset)

def data_exploration(dataset, show = 0):
    # Data exploration process (before applying models)
    dataset_all_quant = dataset.loc[:, ~dataset.columns.isin(['track_name', 'track_artist', 'track_id', 'key', 'genres', 'danceability_binary'])].copy()
    dataset_with_dance = dataset.loc[:, ~dataset.columns.isin(['track_name', 'track_artist', 'track_id', 'key', 'genres', 'danceability'])]
    dataset_with_key = dataset.loc[:, ~dataset.columns.isin(['track_name', 'track_artist', 'track_id', 'genres'])].copy()

    for s in show:
        if s == 1:
            sns.countplot(x = 'danceability_binary', data = dataset, palette=['blue', 'orange'])
            plt.title("Absolute Frequency of Danceability (Binary) variable")
            plt.show()
        elif s == 2:
            pd.crosstab(dataset.key, dataset.danceability_binary).plot(kind = 'bar')
            plt.title('Danceability Frequency for Key value')
            plt.xlabel('Key')
            plt.ylabel('Danceability')
            plt.show()
        elif s == 3:
            plt.scatter(x = dataset.loudness, y =dataset.danceability_binary)
            plt.show()
        elif s == 4:
            sns.pairplot(dataset_with_key, vars = ['energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'], hue = 'danceability_binary')
            plt.title("Pairplot for all the variables")
            plt.show()
        elif s == 5:
            corr_matrix = dataset_all_quant.corr()
            sns.heatmap(corr_matrix, annot= True)
            plt.title("Correlation matrix for all the variables")
            plt.show()
        elif s == 6:
            ### Analyze outliers from the data 
            fig, ax = plt.subplots(3,3)

            for col, val in enumerate(dataset_all_quant):
                i,j = divmod(col, 3)
                sns.boxplot(x = dataset_all_quant[val], ax = ax[i,j])
            plt.subplots_adjust(wspace=0.5, hspace=1)
            fig.suptitle("Distribution of the variables")
            plt.show()
        elif s == 7:
            fig, ax = plt.subplots(3,3, figsize = (11.7, 8.27))
            for col, val in enumerate(dataset_with_dance):
                i,j = divmod(col, 3)
                sns.boxplot(x = dataset_with_dance['danceability_binary'], y = dataset_with_dance[val], ax = ax[i,j], hue = dataset_with_dance['danceability_binary'], legend = False)
            plt.subplots_adjust(wspace=0.5, hspace=1)
            fig.suptitle("Distribution of the variables over danceability")
            plt.show()
        elif s == 8:
            sns.boxplot(x = dataset_with_key['key'], y = dataset_with_key['danceability'])
            plt.title("Danceability over the song's key")
            plt.show()