"""
Data Loading and Preprocessing Module

This module provides functionality to fetch the UCI Adult Income dataset from OpenML.
It includes a robust preprocessing pipeline using scikit-learn Transformers and 
a synthetic data fallback mechanism to ensure the experimental pipeline remains 
executable even without an active network connection.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Loads the UCI Adult Income dataset and applies a standardized preprocessing pipeline.

    The function attempts to retrieve the dataset from OpenML. If the request fails, 
    it generates a synthetic dataset with similar class imbalances to maintain 
    consistent experimental conditions. Preprocessing includes median imputation 
    for missing values, standard scaling for continuous features, and one-hot 
    encoding for categorical variables.

    Args:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator for reproducibility.

    Returns:
        tuple: A tuple containing:
            - X_train_processed (np.ndarray): Preprocessed training features.
            - X_test_processed (np.ndarray): Preprocessed testing features.
            - y_train (np.ndarray): Training labels.
            - y_test (np.ndarray): Testing labels.
            - continuous_indices (list): List of column indices for continuous features.
            - preprocessor (ColumnTransformer): The fitted transformer object.
    """
    try:
        print("Requesting UCI Adult Income Dataset (OpenML ID: 1590)...")
        # auto parser is used to handle categorical types natively in recent sklearn versions
        adult = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        X = adult.data
        # We convert the target to binary integers to facilitate metric calculations like Brier Score
        y = (adult.target == '>50K').astype(int) 
        
        # Categorical and continuous features require distinct transformation strategies
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        continuous_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
    except Exception as e:
        # Fallback is necessary to allow offline development and testing
        warnings.warn(f"Dataset fetch failed: {e}. Utilizing synthetic classification data.")
        X, y = make_classification(
            n_samples=10000, 
            n_features=14, 
            n_informative=10, 
            n_redundant=4,
            random_state=random_state,
            weights=[0.76, 0.24] 
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(14)])
        y = pd.Series(y)
        categorical_cols = []
        continuous_cols = X.columns.tolist()

    # Median imputation is used for robustness against outliers in continuous features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # handle_unknown='ignore' ensures the encoder doesn't crash on unseen categories in the test set
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Stratified split ensures that the class distribution is preserved across subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fitting on train and transforming test prevents data leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # We identify continuous indices because covariate shift simulation specifically targets them
    num_continuous = len(continuous_cols)
    continuous_indices = list(range(num_continuous))

    return X_train_processed, X_test_processed, y_train.values, y_test.values, continuous_indices, preprocessor
