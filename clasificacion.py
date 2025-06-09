import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def random_forest_classifier(x_data, y_data):
    """
    Train a Random Forest Classifier and return the trained model.

    Parameters:
    x_data (pd.DataFrame): Features for training.
    y_data (pd.Series): Target variable for training.

    Returns:
    RandomForestClassifier: The trained Random Forest model.
    """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_data, y_data)

    # Optionally, you can evaluate the model using cross-validation
    scores = cross_val_score(rf, x_data, y_data, cv=5)
    print(f"Cross-validation scores: {scores}")

    return rf

def knn_classifier(x_data, y_data, n_neighbors=5):
    """
    Train a K-Nearest Neighbors Classifier and return the trained model.

    Parameters:
    x_data (pd.DataFrame): Features for training.
    y_data (pd.Series): Target variable for training.
    n_neighbors (int): Number of neighbors to use in the KNN algorithm.

    Returns:
    KNeighborsClassifier: The trained KNN model.
    """
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_data, y_data)

    # Optionally, you can evaluate the model using cross-validation
    scores = cross_val_score(knn, x_data, y_data, cv=5)
    print(f"Cross-validation scores: {scores}")

    return knn

def regression_model(x_data, y_data):
    """
    Train a regression model (e.g., Linear Regression) and return the trained model.

    Parameters:
    x_data (pd.DataFrame): Features for training.
    y_data (pd.Series): Target variable for training.

    Returns:
    LinearRegression: The trained regression model.
    """
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    reg.fit(x_data, y_data)

    # Optionally, you can evaluate the model using cross-validation
    scores = cross_val_score(reg, x_data, y_data, cv=5)
    print(f"Cross-validation scores: {scores}")

    return reg

if __name__ == "__main__":
    file = "data/TR_starPep_AB_training.fasta_AAC_class.csv"
    data = pd.read_csv(file)
    column_class_label = 'Class'
    X = data.drop(columns=[column_class_label])
    y = data[column_class_label]

    # Train a Random Forest Classifier
    rf_model = random_forest_classifier(X, y)

    knn = knn_classifier(X, y, n_neighbors=5)

    reg = regression_model(X, y)