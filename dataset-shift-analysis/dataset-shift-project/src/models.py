"""
Model Definition and Training Orchestration Module

This module defines the architectural suite of classical machine learning models 
used for benchmarking. It provides factory methods for model instantiation with 
reproducibility controls and a unified training loop that handles model fitting 
and exception management during the learning phase.
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier

def get_models(random_state=42):
    """
    Instantiates a diverse collection of classical machine learning models.

    The selection includes models with various inductive biases (e.g., linear models, 
    kernel methods, and tree-based ensembles) and an explicit naive baseline 
    (DummyClassifier). This provides a comprehensive cross-section of algorithmic 
    robustness and a point of comparison for naive performance.

    Args:
        random_state (int): The seed used for reproducible model initialization.

    Returns:
        dict: A dictionary mapping canonical model names (str) to un-fitted 
              scikit-learn model instances.
    """
    models = {
        "Naive Baseline": DummyClassifier(strategy="most_frequent"),
        "Naïve Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_state, algorithm="SAMME")
    }
    return models

def train_models(models, X_train, y_train):
    """
    Executes the training phase for a provided collection of machine learning models.

    This function iterates through the model repository and fits each instance 
    to the training data. It includes per-model exception handling to prevent 
    divergent models (e.g., convergence failures) from aborting the entire 
    experimental pipeline.

    Args:
        models (dict): Mapping of model names to un-fitted model instances.
        X_train (np.ndarray): The training feature matrix.
        y_train (np.ndarray): The target label vector.

    Returns:
        dict: A dictionary containing only the models that were successfully trained.
    """
    print("Beginning model training. Note: Support Vector Machine (SVM) fitting "
          "runtime is significantly higher than other models in this suite.")
    trained_models = {}
    for name, model in models.items():
        print(f"Fitting {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            # We log failures instead of raising to allow remaining models to be evaluated
            print(f"Critical failure during training of {name}: {e}")
    print("Model training phase complete.")
    return trained_models
