from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from scipy.stats.stats import pearsonr

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

def analyze_dataframe(features, filename, graph=0):
    features = dataset_information(dataset=features, filename=filename, change=0)

    # dataset=features[features["user_id"] == "11133022471"]

    data_analysis(dataset=features[features["user_id"] == "Rafa"], show=[4,5,6,7])

    ### Hypothesis Testing

    #testing_hypothesis(dataset = features)


    # Logistic Regression with just one explanatory variable (numeric for now)

    return

    y_test, y_pred, X_test, model = apply_logistic_regression_model(
        X=features.loc[:, features.columns.isin(["energy"])],
        y=features.loc[:, features.columns.isin(["danceability_binary"])],
    )

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

    # sigmoid_function(model.coef_, model.intercept_)

    return

    # Logistic Regression with all the variables. It's so random

    y_test_wo_dummies, y_pred_wo_dummies = apply_logistic_regression_model(
        X=new_features.loc[
            :,
            ~new_features.columns.isin(
                ["track_name", "track_artist", "key", "genres", "danceability_binary"]
            ),
        ],
        y=new_features.loc[:, new_features.columns.isin(["danceability_binary"])],
    )

    print("Accuracy: ", accuracy_score(y_test_wo_dummies, y_pred_wo_dummies))
    print(
        "Confusion Matrix: \n", confusion_matrix(y_test_wo_dummies, y_pred_wo_dummies)
    )
    print(
        "Classification Report:\n",
        classification_report(y_test_wo_dummies, y_pred_wo_dummies),
    )

    # Correlations between variables and danceability
    features_no_track_name = features.loc[
        :, ~features.columns.isin(["track_name", "track_artist", "key", "genres"])
    ]
    correlations = features_no_track_name.corr()
    # print(correlations['danceability_binary'])

    """print(features.head())
    print(features.describe().round(3))
    print(features[features.valence == features.valence.min()]) # Min valence
    print(features[features.valence == features.valence.max()]) # Max valence
    print(new_features.loc[:, ~new_features.columns.isin(['track_name', 'track_artist', 'key', 'genres'])].groupby('danceability_binary').mean())
"""

def dataset_information(dataset, filename, change=0):
    dataset.isnull().sum()  # There are no null values in the data
    dataset.track_name.nunique()  # There are 487 unique values, but the dataset has 527 songs. This means that there are duplicates on the track_name column

    print(dataset.describe())

    ### Check if there are duplicate values on the track_name column
    dataset.loc[dataset.duplicated()]
    # use dataset.query('track_name' == 'Name of the track') to check the duplicates
    dataset[dataset.track_name.duplicated()]["track_name"].tolist()

    print("Dataset shape before duplicates' removal", dataset.shape)
    dataset = dataset.drop_duplicates(subset=["track_name"], keep="last").copy()
    print("Dataset shape after duplicates' removal", dataset.shape)

    ### Change user_ids
    for idx, user_id in enumerate(dataset["user_id"].unique()):
        if idx == 0:
            continue
        if idx == 1:
            dataset.loc[dataset["user_id"] == user_id, "user_id"] = "Figas"
        elif idx == 2:
            dataset.loc[dataset["user_id"] == user_id, "user_id"] = "Pitex"
        elif idx == 3: 
            dataset.loc[dataset["user_id"] == user_id, "user_id"] = "Rafa"
        elif idx == 4: 
            continue

    ### Handle outliers

    columns_to_handle = ["instrumentalness", "loudness", "speechiness", "liveness"] 

    #### Analyze instrumentalness variable

    dataset_instrumentalness = dataset['instrumentalness'].copy()

    Q1 = dataset_instrumentalness.quantile(0.25)
    Q3 = dataset_instrumentalness.quantile(0.75)
    IQR = Q3 - Q1
    lower_lim = Q1 - 0.8 * IQR
    upper_lim = Q3 + 1.5 * IQR
    outliers_15_low = dataset_instrumentalness < lower_lim
    outliers_15_up = dataset_instrumentalness > upper_lim

    dataset_all_outliers = dataset_instrumentalness[dataset_instrumentalness > outliers_15_up]

    """sns.boxplot(dataset_instrumentalness)
    plt.title(f'Distribution of Instrumentalness variable')
    plt.show()
    sns.boxplot(dataset_all_outliers)
    plt.title(f'Distribution of Instrumentalness variable without outliers')
    plt.show()"""


    

    ## Tukey's rule 
    for idx, col in enumerate(columns_to_handle): # This has to be done one by one
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        print(f"{col} -> Q1 = {Q1} | Q3 = {Q3}")
        IQR = Q3 - Q1
        print(f"IQR -> {IQR}")

        lower_lim = Q1 - 0.8 * IQR
        upper_lim = Q3 + 1.5 * IQR

        outliers_15_low = dataset[col] < lower_lim
        outliers_15_up = dataset[col] > upper_lim

        ### Remove outliers above the upper_lim and below lower_lim

        #dataset_tukey = dataset[(dataset[col] < outliers_15_up + outliers_15_up*0.20) & (dataset[col]> outliers_15_low + outliers_15_low*0.20)]
        #print(f"{col} Lower = {lower_lim} | Upper = {upper_lim} need to plot this")

    if change == 1:
        ### Save new csv file
        dataset.to_csv(filename, mode="w", index=False, header=True)

    return dataset

def testing_hypothesis(dataset):

    ### Correlation test -> Check significance over the correlations between all the variables

    pearson_test = {}
    corr_matrix = dataset.drop(columns = 'danceability_binary').select_dtypes(
        exclude=[np.object_]
    ).corr()

    # All thep- values are really low...

    threshold = 0.01
    corr_above_thresh = [[i, j] for i,j in zip(*np.where(np.abs(corr_matrix.values) > threshold)) if i!=j]

    print(corr_above_thresh)
    for idx in corr_above_thresh:
        pearson_test[f'{corr_matrix.index[idx[0]]} and {corr_matrix.index[idx[1]]}'] = \
        pearsonr(dataset[corr_matrix.index[idx[0]]], dataset[corr_matrix.index[idx[1]]])

    print(pearson_test)


def data_analysis(dataset, show=0):
    dataset_no_objects = dataset.select_dtypes(
        exclude=[np.object_]
    )  # Select all columns that are not objects

    # Data exploration process (before applying models)

    # Posso analisar por playlist, agora que tenho as playlists_id (ate posso ir buscar o nome da playlist)

    for s in show:
        if s == 1:
            sns.countplot(
                x="danceability_binary", data=dataset, palette=["blue", "orange"]
            )
            plt.title("Absolute Frequency of Danceability (Binary) variable")
            plt.show()
        elif s == 2:
            pd.crosstab(dataset.key, dataset.danceability_binary).plot(kind="bar")
            plt.title("Danceability Frequency for Key value")
            plt.xlabel("Key")
            plt.ylabel("Danceability")
            plt.show()
        elif s == 3:
            plt.scatter(x=dataset.loudness, y=dataset.danceability_binary)
            plt.show()
        elif s == 4:
            sns.pairplot(
                dataset,
                vars=[
                    "danceability",
                    "energy",
                    "loudness",
                    "speechiness",
                    "acousticness",
                    "instrumentalness",
                    "liveness",
                    "valence",
                    "tempo",
                ],
            )
            plt.show()
        elif s == 5:
            corr_matrix = dataset_no_objects.drop(
                columns=["danceability_binary"]
            ).corr()
            print(corr_matrix)
            sns.heatmap(corr_matrix, annot=True)
            plt.title("Correlation matrix for all the variables")
            plt.show()
        elif s == 6:
            ### Analyze outliers from the data
            fig, ax = plt.subplots(3, 3)
            for col, val in enumerate(
                dataset_no_objects.drop(columns=["danceability_binary"])
            ):
                i, j = divmod(col, 3)
                sns.boxplot(
                    x=dataset_no_objects.drop(columns=["danceability_binary"])[val],
                    ax=ax[i, j],
                )
            plt.subplots_adjust(wspace=0.5, hspace=1)
            fig.suptitle("Distribution of the variables")
            plt.show()
        elif s == 7:
            fig, ax = plt.subplots(3, 3, figsize=(11.7, 8.27))
            for col, val in enumerate(
                dataset_no_objects.drop(columns=["danceability_binary"])
            ):
                i, j = divmod(col, 3)
                sns.histplot(
                    x=dataset_no_objects.drop(columns=["danceability_binary"])[val],
                    legend=False,
                    ax=ax[i, j],
                )
            plt.subplots_adjust(wspace=0.5, hspace=1)
            fig.suptitle("Distribution of the variables")
            plt.show()
        elif s == 8:
            for idx, val in enumerate(
                dataset_no_objects.drop(columns=["danceability_binary"])
                ):
                sns.boxplot(x=dataset["key"], y=dataset[val])
                plt.title(f'{val} over the song\'s key')
                plt.show()


def apply_logistic_regression_model(X, y):
    # Logistic Regression

    # Basic framework for logistic regression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(solver="liblinear", random_state=0)
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)

    return (y_test, y_pred, X_test, model)
