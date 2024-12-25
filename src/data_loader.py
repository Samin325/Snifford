import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(dataset_folder):
    """
    Loads and preprocesses the data

    Assumes data is found in a folder, is in csv format, and has same columns across files
    """
    csv_files = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]
    data = {}

    for csv_file in csv_files:
        file_path = os.path.join(dataset_folder, csv_file)
        df = pd.read_csv(file_path)

        # get rid of leading/trailing whitespace in column names
        df.columns = df.columns.str.strip()

        # change labels so the classifications are binary
        df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

        # extract features and labels
        X = df.drop(columns=['Label'])
        y = df['Label']

        # handle infinity and nan values by replacing with mean
        if X.isin([np.inf, -np.inf]).any().any():
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.mean(), inplace=True)

        # normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        data[csv_file] = (X_scaled, y.values)

    return data


def split_data(data):
    """
    Splits the dataset into training and testing sets
    """
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

    # perform the train/test split on each csv file in the data folder and add the splits to a list
    for _, (X_scaled, y_encoded) in data.items():
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    # concatenate the lists to have one train/test split across all files
    X_train_all = np.concatenate(X_train_list, axis=0)
    X_test_all = np.concatenate(X_test_list, axis=0)
    y_train_all = np.concatenate(y_train_list, axis=0)
    y_test_all = np.concatenate(y_test_list, axis=0)

    return X_train_all, X_test_all, y_train_all, y_test_all
