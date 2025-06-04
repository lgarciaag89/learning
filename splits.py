import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score


def simple_train_test_split(X, y, test_size=0.2, random_state=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def stratified_train_test_split(X, y, test_size=0.2, random_state=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)



if __name__ == "__main__":
    path = "data/TR_starPep_AB_training.fasta_AAC_class.csv"
    column_class_lavel = 'Class'

    data = pd.read_csv(path)

    X = data.drop(columns=[column_class_lavel])
    y = data[column_class_lavel]


    # Perform a simple train-test split
    X_train, X_test, y_train, y_test = simple_train_test_split(X, y, test_size=0.2, random_state=42)
    print("Simple Train-Test Split:")
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Perform a stratified train-test split
    X_train_stratified, X_test_stratified, y_train_stratified, y_test_stratified = stratified_train_test_split(X, y, test_size=0.2, random_state=42)
    print("\nStratified Train-Test Split:")
    print(X_train_stratified.shape, y_train_stratified.shape)
    print(X_test_stratified.shape, y_test_stratified.shape)